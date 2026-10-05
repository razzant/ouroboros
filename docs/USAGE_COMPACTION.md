# Usage-ledger compaction and monetary authority

Terminal usage history is compacted into a stamped baseline block while the
exact original ledger bytes remain in an append-only archive. The ledger is
the monetary authority, so compaction preserves exact money, attribution,
in-flight attempts and historical joins.

## 1. Problem

`state/usage_attempts.jsonl` is the append-only monetary authority
(`ouroboros/usage_ledger.py`). Historical full replay under its monetary lock
caused contention as the file grew. Current writers incrementally validate and
account touched attempts; cold/replaced generations prepare outside the lock
and reconcile under it. `USAGE_LEDGER_WARN_BYTES` in `ouroboros/context_budget.py`
still warns about retained history and maintenance cost. Full-record maintenance
and exceptional corruption repair can still require a complete locked replay. Without compaction the live file grows without bound: each
physical attempt appends a 2–4 row lifecycle chain that remains after it is terminal.

## 2. Compacted representation

Fold the terminal history into a **stamped baseline block** at the head of the
ledger and move the raw pre-compaction bytes, verbatim, into an append-only
archive segment. Nothing is deleted: the archive holds every original row
(with its original `seq`); the live file holds an exact, structurally
validated aggregate plus every row that is still live.

## 3. What folds, what never folds

| rows | disposition | why |
|---|---|---|
| `kind="attempt"`, final state `settled` / `released`, no abandonment marker, and **no review attribution** (`review_skill`/`review_wave_id`/`review_slot_id` all empty) | folded (their whole seq chain) after the fold horizon | ordinary terminal rows are immutable; aggregation-complete under §5 |
| `kind="attempt"`, final state `reserved` / `dispatched` / `unresolved`, or an administrative `settled` row with `settle_reason="abandoned"` | **retained verbatim** | their next valid transition or late receipt still has to join the exact `attempt_id` in the live replay |
| `usage_baseline` / `usage_baseline_group` from a previous compaction | re-folded (header replaced, groups merged by key, exact-decimal sums added) | baselines must not accumulate per epoch; historical unknown-cost groups remain valid aggregates, never recreated individual attempts |
| `kind="subscription_session"`, `"external_unmetered"` | **retained** | their `attempt_id` is deterministically re-derived from a stable external id and re-asserted on replay: `_append_single_settled_row` dedups and conflict-checks against the LIVE replay. Folding them would turn an idempotent replay into a silent double charge. Disclosed residual: these rows keep growing (slowly — one row per delegated run / external dispatch). |
| `kind="legacy_*"` | **retained** | same idempotency argument: `ensure_legacy_imported` dedups candidate rows against live `attempt_id`s if the completion watermark is ever lost mid-history. Bounded one-time set. |
| attempts with review attribution | **retained** | `skill_review_usage` projects historical waves per-attempt (`attempt_ids`, `attempts` lists) for durable review receipts; folding would erase that projection. Disclosed residual (skill-review waves only; ordinary task/review traffic carries no `review_*` attribution). |
| unknown future kinds | **retained** | fail-safe default: fold only what this design proves aggregation-complete |
| quarantine file | untouched | quarantined bytes are already out of the replay; `integrity_degraded` stays path-derived and unaffected |

An administrative abandonment uses the existing `settled` state with
`settle_reason="abandoned"`, `cost_usd=None` and `cost_final=false`. The reservation
bound remains accounted without becoming confirmed spend; closing ownership does
not make an unknown price final. `usage_ledger.is_abandoned_settlement` is the shared
structural predicate, and validation rejects an abandonment marker that claims a
known price or final cost.

An unresolved attempt can become this administrative settlement or accept one real
settlement tagged `late_receipt`; an abandoned settlement can accept that same late
receipt. Either may instead become `released` only on the existing positive
`before_dispatch_failed:` proof that generation never started. After the real
settlement or release the row is immutable again. Identical actual-receipt retries
are no-op at the accounting writer; conflicting receipts cannot append. The full
validator and incremental `LedgerResumeState.late_receipt_ids` preserve the same
eligibility, so a warm read grants no extra transition and loses no owed receipt.
Compaction preserves the whole eligible chain, not just its last row, including a raw send's optional `local_answer_owner_pid`. Proven local consumer death can change writer classification; it never makes an unresolved monetary chain foldable. Existing
baseline groups keep their recorded sums and weights; the archive is not migrated
to invent individually correctable attempts.

## 4. Baseline block shape

Subscription-session replay binds the stable external session id, route and all
ownership/review-attribution fields. Its observed model is disclosure, not a new
physical session or a pricing input: a later observation can differ or become
unknown while replay returns the original ledger row byte-identically. Existing
legacy model labels are historical evidence, not newly confirmed observations;
new custody/result observations remain separate. External-unmetered replay keeps
its own existing model identity check.

The compacted file is `[header row] [group rows …] [retained rows …]`.

**Header** (`kind="usage_baseline"`, exactly one, always seq 1 when present):

```json
{"kind":"usage_baseline","attempt_id":"baseline-<hex12>","state":"settled",
 "seq":1,"ts":"…","baseline_id":"<hex12>","compaction_epoch":N,
 "archive_rel":"archive/usage_ledger/segment_….jsonl",
 "source_sha256":"…","source_size_bytes":B,"source_row_count":R,
 "source_first_seq":1,"source_last_seq":K,
 "folded_row_count":F,"folded_attempt_count":A,
 "group_count":G,"retained_row_count":T}
```

The header is a pure stamp: `_summary`/`_breakdown_bucket` skip it entirely
(it contributes no money, no counts, no tokens). `source_sha256` is the
SHA-256 of the archived segment's exact bytes — together with the segment's
own embedded previous header this forms a tamper-evident hash chain over the
whole history.

**Group rows** (`kind="usage_baseline_group"`): one per attribution tuple

```
key = (state, model, provider, category, source,
       task_id, root_task_id, parent_task_id, billing_group_id,
       has_group_limit, group_limit, group_limit_source, group_limit_revision,
       prompt_cache_ttl, cost_known, cost_final, pricing_known, bound_known)
```

(`billing_group_id` keeps an owner Continue's whole-work group intact through
the fold; a group row carries it only when non-empty. Cap presence, value,
source and revision are preserved together, including an explicitly unbounded
cap, so compaction cannot replace an original group cap with later configuration.)

carrying: the key fields verbatim; `folded_attempt_count` (int ≥ 1);
`cost_usd` / `reservation_upper_bound_usd` as **exact-decimal JSON strings**
(§5); token sums (`prompt_tokens`, `completion_tokens`, `cached_tokens`,
`cache_write_tokens`) as ints, absent when no folded row reported the field;
`root_limit_usd` = min over the group's known values (else absent);
`baseline_id` joining the header; empty `review_*` attribution.

Why per-group rows rather than a single global aggregate:
budget enforcement is **per-root** (`reserve_attempt` filters finals by
`root_task_id`; `usage_projection` takes `min` of row `root_limit_usd`), and
`usage_breakdown` groups by model/provider/category/task/root. A single global
row would silently zero per-root accounting — a budget bypass. One stamped
baseline *block* whose group rows preserve the full attribution tuple keeps
every existing aggregation exact while remaining one atomic, stamped unit.

## 5. Monetary exactness rule (the fixed rule)

Monetary equality is defined **on decimals, never on float accumulation**:

- Exact decoding preserves finite accepted monetary scalars,
  including legacy JSON booleans (`true` = 1, `false` = 0) and numeric strings.
  The shared `_usage_money.decimal_of` conversion applies to replay, incremental
  cash and compaction; rejecting a valid historical boolean would erase money.
  Validation and fold eligibility remain with their existing owners.
- Nonfinite monetary values (`Infinity`, `-Infinity`, `NaN`, including strings)
  raise `UsageNonFiniteMoney`, a `UsageAccountingError` distinct from torn-row
  corruption. Validation, exact cash decoding and legacy-source import fail
  closed: strict projection/admission and compaction cannot price that evidence
  as zero or drop it through tail quarantine. Existing ledger/archive bytes and
  reservations remain intact; a refused import retains its source archive and
  does not publish a completed watermark. This intentionally narrows the former
  numeric grammar (which accepted positive infinity): the data remains, but
  strict accounting requires repair of the nonfinite source. Finite scalar
  conversion, six-place rounding and counter semantics are unchanged.
- The compactor parses the source segment with `parse_float=Decimal` and sums
  each group's `cost_usd` / `reservation_upper_bound_usd` as exact `Decimal`s
  of the literals actually stored in the file.
- Those sums run in an **explicit** decimal context (`_exact_money`:
  `prec = MONEY_PRECISION = 60`, `Inexact` trapped), never the ambient one.
  The interpreter default keeps 28 significant digits, which silently rounds
  a large-magnitude sum — and because the group row and the self-check that
  approves it are computed the same way, a rounded total verifies against
  itself while the float projection stays equal on both sides. Sixty digits
  is far past any real ledger; past even that, the trap turns the loss into
  an abort instead of an approximation. Decimal *construction* is
  context-free by language rule, so stored literals are always captured
  exactly; only arithmetic needs the context.
- Group sums are stored as exact-decimal **JSON strings** (`"cost_usd":
  "12.3456789"`, `format(dec, "f")`, no exponent). Validation accepts numeric
  strings; `_usage_money` consumes them directly as Decimal, without float
  conversion. Ordinary public row/projection shapes remain compatible.
  A float re-serialization would round the exact sum to the nearest double;
  the string keeps the invariant byte-checkable forever, including across
  re-compactions (group merges sum the decimal strings exactly).
- Retained rows are re-serialized from the standard float parse (shortest-repr
  round-trip). The compactor VERIFIES, per retained row, that the re-parsed
  `Decimal` view of the new line equals the `Decimal` view of the original
  line on every field (only `seq` / `pre_compaction_seq` may differ); every
  literal our writers ever emit is shortest-repr so this holds identically,
  and a foreign non-canonical literal (one that is not double-round-trippable)
  triggers **abort, not approximation**.
- Before committing, the compactor replays the candidate bytes through the
  PRODUCTION aggregation (`_final_rows` → `_summary`, per-root summaries with
  `min` limits, `_breakdown_bucket` global and per model/provider/category/
  task/root axis) and requires the **non-money** rendered dicts to equal the
  corresponding source projection. The separate `decimal_totals` check
  requires exact money equality. Either mismatch aborts the compaction and
  leaves the ledger byte-identical; float accumulation is not an additional
  equality gate.

The stored decimal money and non-money projections are preserved before the
swap. Full replay and incremental cash now share exact reversible Decimal
contributions under the same precision-60/Inexact-trapping context. Each cash
bucket rounds half-even to six places before the accounted sum, as admission
requires; its `1e-9` allowance remains. This deliberately corrects the old
float ordering artifact: exact `1.9542475 + 0.513341 = 2.4675885` renders
`2.467588` both before and after compaction, instead of occasionally `2.467589`.
Historical charge literals remain unchanged. The independent compaction money
self-check remains required; agreement between two renderers is not proof of
exactness. Raw numeric monetary literals retain precision across transitions.

## 6. seq policy

The live file keeps the substrate's strongest integrity property: **dense
`seq` from 1** (the validator's density check is what detects mid-file loss
and tampering). A gap-tolerant validator would have weakened that authority,
so instead the compacted file starts a fresh dense epoch:

- baseline header = seq 1, groups 2..G+1, retained rows follow densely in
  their original relative order;
- every retained row keeps its original seq as `pre_compaction_seq`;
- the header records `source_first_seq`/`source_last_seq`, and the archive
  segment holds every original row with its original `seq` untouched.

Monotonicity and density are preserved; the original seq
values are never lost (archive + `pre_compaction_seq`). Nothing durable
references ledger rows by `seq` (cross-references are `attempt_id`s); resume
fingerprints are invalidated structurally by the inode change (§8).

Validator additions (`_validate_records`): baseline rows are legal only as
the leading block of a full-file validation (a baseline row in an appended
tail, or after any non-baseline row, is corrupt); exactly one header, first;
group rows must carry the header's `baseline_id` and a positive
`folded_attempt_count`; group state ∈ {settled, unresolved, released}, including
historical unresolved groups. Per-attempt transitions follow the late-receipt
contract in §3; numeric checks accept monetary strings through `_number`.

**The stamp's own provenance is validated, not trusted.** The header is the
only ledger row that points at bytes outside its file, so the substrate — the
thing that decides what a well-formed row IS — checks it:

- `compaction_epoch` is a positive int; `source_sha256` is 64 hex;
- `archive_rel` is *bounded* (`usage_ledger.valid_archive_rel`): relative,
  forward-slash, exactly `archive/usage_ledger/<name>`, no `..`, no absolute
  path, no drive letter, no separator inside the name. A tampered header can
  therefore never aim a reader at a file elsewhere on the host;
- the counts must **close**: `folded_row_count + retained_row_count ==
  source_row_count`, `source_first_seq == 1`, `source_last_seq ==
  source_row_count`, and `source_size_bytes ≥ 1`;
- the block must agree with its header: the number of group rows equals
  `group_count` and their `folded_attempt_count`s sum to the header's. Once
  the first non-baseline row closes the block, a later group row — even one
  carrying the real `baseline_id` — is corrupt. That is the money-injection
  shape the position rule exists to refuse;
- `pre_compaction_seq` is checked as the provenance claim it is: legal only
  under a leading header, strictly increasing, only up to the first
  non-baseline row that carries none (post-compaction appends never carry
  it, and a re-compaction rewrites the whole file), and **inside the source
  range the header declares** (`source_first_seq`..`source_last_seq`) — a
  retained row may not claim to come from bytes nobody archived.

## 7. Aggregation contract (`_usage_rows`)

`_summary` and `_physical_call_count`/`_breakdown_bucket` are
baseline-aware in the narrowest way:

- `usage_baseline` header: skipped (no money, no counts);
- `usage_baseline_group`: every **count** increment (`attempt_counts`,
  `unknown_unmetered`, `non_final_rows`, physical calls, `prompt_cache_ttls`)
  uses `weight = folded_attempt_count`; every **sum** adds the row's carried
  aggregate once. For all existing kinds `weight == 1` and the code path is
  unchanged.

The group key (§4) makes each group homogeneous in every branch predicate
`_summary` evaluates per row (`cost is None`, `cost_final`,
`pricing_known is False`, `bound is None`, state), so the per-group branch is
exactly the per-row branch taken `weight` times with the sums pre-added.

Original cap authority is separate from aggregate minima. `BindingIndex`
keeps the earliest original root/group fields verbatim, including explicit
unlimited `None`. Compaction stamps `binding_authority=carried` and puts
`original_root_binding` / `original_group_binding` on the first aggregate for
each identity. Exact `unbound` leaves that axis open to the first later original
row.

Nothing is recovered from the archive. A block row that carries no usable
binding — an unstamped header (a pre-carriage pass), a header stamped
`unknown`, a missing, `unknown`, malformed or foreign carriage — binds its
member from the row's OWN cap literal (`billing_group_limit_source=legacy_live`,
disclosed through `ledger_billing_binding` / `original_group_limit`) while
every block row of that member that carries a cap carries the same one (a
group's rows must agree on the group cap; their member roots' own caps may
differ; a row without a cap carries no literal and does not vote, so neither
row order nor the moment of compaction changes the answer). Disagreeing
literals leave the member open (`no_attempt_recorded`): admission then
binds the root as its own group under the configured cap and discloses that on
the task (`legacy_default`: `usage_admission.task_billing_fields`,
`continuation_admission._billing_group`). Accumulated spend is preserved either
way, and the next pass carries the literal binding — or the open member —
forward. A binding pinned on the task result and owner amendments remain
separate, prior authorities.

The money path never reads an archive segment: strict writer preparation
parses the live ledger only, display and raw-record reads the same, and
compaction compares exact source/candidate bindings and money on the live
rows (the pass still writes its own new segment). The archive chain stays
available for explicit history questions (§10) and audits.

## 8. Concurrency, crash-safety, caches

- Compaction runs **only under the same monetary lock** (`_locked(root)`) as
  every read-check-append transaction, invoked from `reserve_attempt`'s
  locked section before its ledger read (§9). No second lock, no new lock
  ordering.
- **The lock has two tiers, chosen by a capability predicate, and the pass
  runs on one of them only.** `platform_layer.kernel_file_locks_enforced`
  decides once per lock directory by kernel-locking a scratch file there:
  only the kernel's own "this filesystem cannot" selects the **name tier**;
  every other answer is the **enforced tier**. The tier is never chosen by a
  refusal on a live acquisition. The refusals that mean the filesystem takes
  no kernel locks at all are exactly `EOPNOTSUPP`/`ENOTSUP`/`ENOSYS` — on
  Windows `ERROR_INVALID_FUNCTION` and `ERROR_NOT_SUPPORTED`, which
  `_win32_lock_error` maps onto `ENOSYS` and `EOPNOTSUPP`, because CPython's
  own winerror→errno table lands both on `EINVAL` and would leave the name
  tier structurally unreachable there: a Windows volume without byte-range
  locks would fail EVERY monetary append closed instead of degrading to it.
  `ENOLCK` ("no locks available" — a filesystem without a lock daemon, or an
  exhausted kernel lock table) selects the **name tier** too. Making this
  failure close every caller would stop locked state writes, task-result and
  custody updates, and model dispatch on a lockd-less filesystem, although
  those non-monetary consumers can use the name protocol. The probe records
  the errno beside its verdict, and each caller can refuse that tier by errno
  (`acquire_exclusive_file_lock(refuse_name_tier_errnos=…)`).
  Only the monetary lock does: `usage_ledger._named_lock` names `ENOLCK`, so on
  such an install every monetary write refuses typed (`UsageAccountingError` —
  no lock, no append, no pass; money never runs the name protocol where locks
  might merely be unavailable) while every other lock keeps the name protocol
  it always ran there; moving `state/` onto a filesystem that locks is the
  repair. The verdict is cached per process per directory under one module
  lock, so racing threads share one probe and one answer rather than two
  probes that could disagree — except a directory where the scratch probe
  cannot be created, which answers enforced for that call and is probed again
  next time (not cached).
  **Windows takes the enforced tier on a byte range beyond the stamp.**
  Windows byte-range locks are mandatory: locking the whole file prevents a
  contender from reading the owner's stamp, so it cannot judge the hold and
  times out. Falling back to the name tier would expose another constraint:
  the contender's open handle can prevent the owner's release unlink through
  a sharing violation (no `FILE_SHARE_DELETE`). `_unlink_lock_path` retries
  that transient refusal for a bounded window. The enforced-tier hold is one
  byte at
  `platform_layer._WIN32_LOCK_OFFSET` (`0x7FFFFFFF00000000`, length 1 — the
  common Win32 idiom; a lock beyond end-of-file is legal there and no lock
  file's one-line stamp can reach that far), so the bytes a contender reads,
  [0, 512), lie outside every range this protocol locks. The capability probe
  then runs on Windows exactly as on POSIX (scratch file, lock, unlock;
  ERROR_LOCK_VIOLATION = held, ERROR_INVALID_FUNCTION/ERROR_NOT_SUPPORTED = no
  byte-range locks on this volume → the name tier), and the compaction pass
  runs there. Windows eviction on this tier takes the SAME non-blocking kernel
  lock on the judged descriptor as POSIX — a creator stalled between its create
  and its lock is judged by the kernel, not by age alone — but it cannot unlink
  what it holds open, so it releases that hold, closes its probe and only then
  re-checks the identity and unlinks. It therefore does NOT give the POSIX
  guarantee below («of two racing reclaimers at most one can evict», held by
  the kernel across the whole judge → re-check → unlink span): two Windows
  reclaimers may both re-check, and what keeps the shape exclusive across that
  gap is Windows itself — the winner's freshly won lock is held open by its
  owner, so the loser's unlink is REFUSED rather than obeyed (a sharing
  violation the eviction path deliberately does not retry; the poll loop
  re-judges). Release order there is unlock → close → unlink (a handle closed
  with an outstanding lock leaves the release undefined; the unlink still
  retries a contender's transient refusal). What Windows still does not have on
  this tier is named elsewhere in this section and unchanged: no directory
  fsync, and no old-inode witness across `os.replace`, so a charge landed in
  the swap's last syscall is lost silently there. Windows execution is checked
  by the CI matrix; host-side pins (the range constant and its two wrappers, an emulated LockFileEx that
  refuses the same range, the delete-semantics simulator) stand in for the
  mechanism, never for the platform.
  *Enforced tier* (POSIX `fcntl.flock`, Windows `LockFileEx` —
  both held on the lock fd): every other hold of this lock is milliseconds; a compaction
  pass over a multi-megabyte ledger can legitimately exceed its 90 s
  staleness window, and a lock evicted purely by age would put a second
  writer on the same authority. So the acquisition takes a **kernel lock on
  the lock fd** in addition to the O_EXCL name. The refusals that mean HELD
  BY SOMEONE — stand down and re-contend — are exactly `EAGAIN`/`EWOULDBLOCK`
  (on Windows `ERROR_LOCK_VIOLATION` alone, mapped onto the first); every
  other refusal, `EACCES` included, **fails closed** — no descriptor, our own
  file removed, a stale lock never evicted without the held flock, never a
  silent fall back to the name protocol — `_named_lock` acquires
  **owner-aware** (a live owner PID is never evicted on age), and the hold
  yields a **heartbeat**
  (`platform_layer.refresh_exclusive_file_lock`), REQUIRED by both entry
  points rather than defaulted — an omitted heartbeat would turn every proof
  below into a no-op and swap the authority unproven, so a caller that drops
  it gets a TypeError — and the pass beats at every checkpoint — inside the long span (both row walks of the candidate build
  and each verification stage), then **immediately before each snapshot
  re-check, including the final one inside the atomic replace** (writing and
  fsyncing the candidate temp can take arbitrarily long, so the proof of
  ownership runs once the temp is durable, right before the rename — ownership
  first, then the snapshot): a re-check answered while the lock already
  belongs to someone else is a meaningless answer, so the loss must abort the
  pass before the answer is even asked. The lock primitives are
  ownership-exact and kernel-guarded: a stale eviction unlinks only while
  HOLDING the flock on the very fd it judged abandoned (path re-checked
  against it), so of two racing reclaimers at most one can evict; on POSIX a
  release unlinks before its close, under the still-held flock; the heartbeat
  answers OWNERSHIP rather than success, including for a lock file replaced
  atomically (the path never absent), and answers `False` — never a renewal —
  when its own descriptor's identity cannot be read or the kernel refuses the
  `utime`. Windows cannot unlink an open file: its eviction takes the same
  probe lock and then re-checks the path after releasing and closing that
  probe, AND its release re-checks after its own close, a freshly won lock —
  held open by its owner — being undeletable there, which is what keeps that
  shape exclusive across the gap the kernel does not cover (above). Disclosed, both tiers: a contention
  answer (`EAGAIN`) on the creator's OWN fresh file — a foreign flock holder
  that never unlinks it — leaves that file on the path, stamped with the
  creator's live pid; the creator re-contends against it until its timeout,
  and owner-aware acquirers never age it out while this process lives. No
  in-protocol holder produces that shape (an evictor's flock unlinks what it
  judged), so it is theoretical on the enforced tier. The acquisition is identity-checked too: a creator
  stalled between its O_EXCL create and its kernel lock (SIGSTOP, a suspend,
  a debugger, clock skew) can be judged abandoned and evicted, and its lock
  then lands on an inode the path no longer names — not a hold: a won lock is
  returned only while the path still names its descriptor, else the creator
  closes it and re-contends, and the owner pid is written BEFORE the lock so
  an owner-aware reclaimer never judges a live creator's fresh file empty
  (every caller of the primitive gets this, the age-only non-monetary locks
  included: at most one descriptor ever answers, the cost is a re-contention).
  A descriptor whose OWN identity cannot be read — `fstat` answering ESTALE
  or EIO on the very filesystems this tier exists for — proves neither
  ownership nor eviction, so it is not a hold either: comparing two unreadable
  identities raw made them equal and returned a descriptor for an unlinked
  inode. It fails closed, and the file we stamped with our LIVE pid is removed
  with it when its bytes are still exactly the ones we wrote — left behind, no
  owner-aware reclaimer could ever evict it and the lock wedges for good.
  **Residual, disclosed:** the owner-aware rule asks
  `pid_is_alive(owner_pid)`, and a recycled PID reads as alive whoever now
  owns it: `kill(0)` succeeds for the same user and answers `EPERM` for
  another user's process; both mean alive. A lock whose owner died and whose
  PID was reused is never reclaimed by age while the impostor lives (`pid_is_alive` is the
  ONE liveness primitive, shared by every consumer — custody settlements,
  claim reclaims, staging reaps — so a pid recycled onto another user's
  process reads alive everywhere and those defer while the impostor lives,
  exactly as they do for a same-uid recycle): the wedge begins once
  the dead owner's file is older than the 90 s staleness window
  (`usage_ledger._locked`, `stale_sec=90.0` — a literal there, not a
  `config.py` constant) and ends when the impostor exits, even though the
  enforced tier's probe flock would settle it at once — deliberately not
  consulted while the pid reads alive, because a mixed-tier name-tier holder
  holds no flock; `state/usage_attempts.lock` needs a hand repair meanwhile
  (`docs/PERSISTENCE.md` says so in its row).
  On this tier we do not claim a pass can never be robbed. The bounded claim:
  a concurrent holder can exist only after the lock file is removed by an
  actor OUTSIDE the lock protocol — a hand repair, a foreign helper, a
  name-tier process of a mixed-tier install — because in-protocol eviction
  is impossible under the held flock and the heartbeat-fresh mtime; such a
  robbery is caught at the next proof, and ownership is proven at every
  checkpoint, immediately before every rename attempt and once more AFTER
  the in-swap snapshot look, so the irreducible residual is the interval
  between that last proof and the rename syscall: a charge the robber lands
  inside it is erased by the swap. A post-swap re-read cannot detect that
  loss: it compares the new inode against the candidate, while the archive
  contains the pre-row snapshot. On POSIX the swap therefore holds the old
  inode open across the rename (the only witness left) and reads whatever
  landed beyond the proven snapshot's length AFTER the fact: those bytes go
  to `state/usage_attempts.quarantine.jsonl` (`raw_base64`, the shape a torn
  tail takes, which flips `integrity_degraded`) and the pass raises
  `UsageLedgerCorrupt` instead of returning a receipt — the charge is still
  gone from the live file (never re-appended: `seq` belongs to the live
  file), but it is preserved, flagged and typed. Detected by size: a
  same-size in-place rewrite inside that one syscall is not a landed charge
  and is not seen. Windows cannot hold the destination open through
  `os.replace`, so there the loss stays silent, disclosed. A `False`
  heartbeat — or one that cannot be answered at all — aborts the pass,
  leaving the ledger byte-identical.
  *Name tier* (a filesystem the kernel says cannot lock — one answering
  `EOPNOTSUPP`/`ENOSYS` to `flock`; a lockd-less NFS answers `ENOLCK` and is
  fail-closed, above): the O_EXCL name protocol runs alone with re-check-then-unlink
  eviction. There is no kernel exclusion, so a heartbeat there is an identity
  check, not a proof. The pass therefore **refuses to run at all** on the
  name tier (`usage_compaction.NAME_TIER_REFUSAL`: logged, and written ONCE
  per process per data root as a typed `usage_ledger_compaction_refused`
  event to `logs/events.jsonl`; the ledger stays uncompacted and the 20 MB
  tripwire names this tier as a cause), while ordinary monetary appends
  continue under the name protocol as a disclosed best effort: on that tier
  the appends are only as exclusive as the name protocol, which is exactly
  why the whole-file rewrite is withheld. Residual, disclosed: the tier is
  decided per process per lock directory, so a lockd dying mid-run can leave
  one process on each tier until restart — the name-tier process never
  compacts, but it evicts by NAME with no kernel hold, so in that mixed mode
  the two-writer class returns for the enforced-tier process's heartbeat-less
  APPENDS (its compaction pass is still guarded by its proofs).
- **The swap re-proves its snapshot — before, at the last instant, and
  after.** Because the swap replaces the WHOLE file, the pass re-reads the
  live ledger under the same held lock before the archive write and again
  before the rename, and the atomic writer evaluates one more re-proof
  **inside the swap** — after the candidate temp bytes are durable,
  immediately before `os.replace`, and again before EVERY retried attempt
  (`utils.replace_atomic` retries a Windows sharing violation with pauses,
  and a proof taken before a refused attempt is stale by the next one), the
  last instant each replace can still be refused, with the ownership
  heartbeat asked FIRST, the snapshot compare only under a proven hold, and
  the heartbeat asked AGAIN after that compare — so a hold lost while the
  temp was written, between two attempts, or during the compare itself
  refuses the replace, a row that lands after the outer re-check or between
  attempts survives too, and the only interval left between the last proof
  and the rename is the syscall, not the milliseconds of a full-file
  compare. The cost of any lost race is one skipped pass, never a charge.
  (Belt to the lock's braces: it also covers a lock broken by a hand repair
  or an older build — up to that last proof; the bounded claim above names
  the residual.) The compare→replace window is closed structurally as well: EVERY
  writer of this ledger appends under this same owner-aware lock — kernel-held
  on the enforced tier, the only tier the pass runs on — and there is no
  unlocked fallback: an acquisition that times out, or that the kernel
  refuses, raises `UsageAccountingError` and writes nothing. After the rename the pass
  re-reads what landed and refuses to report a receipt for bytes that are not
  there.
- Commit order: (1) build + fully verify the candidate in memory (§5, §6);
  (2) write the archive segment — exact source bytes — via O_EXCL write and
  `fsync` the file **and every directory in the chain, on every pass**: the
  segment's parent, `archive/`, and the data root, since syncing a directory
  persists only the entries IT holds. Unconditional, not only where this pass
  created a level: an earlier pass may have created one and died before its
  fsync, so on a retry the directories exist while their durability does not.
  The archive path must also BE this data root's own — a symlink at
  `archive/` or `archive/usage_ledger` aborts the pass, because history must
  never be written through a link — and on POSIX that bound is enforced **at
  the write itself**, not only at a preceding check: the chain
  root→`archive/`→`usage_ledger` is opened `O_DIRECTORY|O_NOFOLLOW`
  handle-to-handle (`dir_fd`), the segment is created `O_NOFOLLOW` relative
  to the held handle, and file plus every directory HANDLE are fsync'd — so
  the chain proven durable is the one the bytes actually landed in, and a
  link planted between check and write is an abort, not a redirect. On POSIX
  a directory-fsync failure **aborts the pass** (an unsynced archive
  directory plus a completed swap is exactly the crash that loses history);
  Windows has no `dir_fd`/`O_DIRECTORY` and no directory handle to fsync, so
  it keeps the path-based chain as a disclosed best effort chosen by the
  platform predicate, never by swallowing `OSError`; (3) atomically replace
  the live ledger (`_write_bytes_atomic_fsync`), which re-proves the snapshot
  one last time before its `os.replace` (the swap bullet above); (4) emit the
  `usage_ledger_compacted` event. A crash before (3) leaves the ledger
  byte-identical (an orphaned archive segment is harmless and disclosed); a
  crash during (3) leaves either the old or the new file — both valid.
- Cache coherence is structural, not cooperative: the swap changes the inode,
  so every resume fingerprint (`_LEDGER_READ_CACHE`, `_ROWS_MEMO`) refuses to
  warm-resume and refolds from the new file. No cache is asked to remember to
  invalidate.

## 9. Trigger policy (config SSOT, no env knob)

- `ouroboros/config.py`: `USAGE_LEDGER_COMPACT_BYTES = 8_000_000` (compact at
  ~0.2 s-per-replay scale, well under the 20 MB measured-degradation WARN) and
  `USAGE_LEDGER_COMPACT_RETRY_GROWTH_BYTES = 1_000_000`. Constants, not env
  handles.
- `reserve_attempt` calls `maybe_compact_usage_ledger_locked(root)` at the top
  of its locked section: an `os.stat` fast-path (~µs) below the threshold;
  above it, one compaction pass on exactly the path whose lock-hold the file
  size degrades. Every failure inside compaction is contained and never fails
  the reservation: a policy abort (`_Abort`) is logged AND records the typed
  `usage_ledger_compaction_skipped` event naming its reason, once per process
  per cause. Disclosed: the two snapshot-integrity abandons (a row that landed
  under the lock between the proven snapshot and the archive write, or between
  that write and the swap) log a warning and return without the typed event,
  because they report a lost race the next pass simply repeats rather than a
  cause an operator has to diagnose. A structurally corrupt ledger still fails
  in the normal read path with the normal error.
- Thrash guard, in two parts, and a pass runs only past both. The threshold
  alone is no brake: the unfoldable residue (group rows, retained idempotent
  and review-attributed rows, terminal rows younger than the fold horizon)
  only grows, so once it reaches the trigger every reservation would run a
  full rewrite of the authority under the held lock and copy the whole live
  file into a new archive segment for a gain of a few kilobytes. Profitability
  is not the question the guard asks; a pass is worth its cost only after real
  growth.
  - **The floor is the ledger's own, so it holds across processes.** The
    header the last committed pass stamped into line 1 records
    `source_size_bytes`, the size that pass READ, and no pass runs until the
    live file reaches it plus `…_RETRY_GROWTH_BYTES`. Read lock-free by
    `_live_baseline_header` (one `readline`; appends never touch line 1 and
    the swap is atomic), so every process answers from the same number. This
    is what a per-process memo cannot do: a committed pass REPLACES the file,
    so at that instant every other process's memo stops matching and re-enters
    a pass, and a freshly started process has no memo at all while the residue
    holds the file above the trigger for good. Measured on the owner's live
    install before this floor: 100 passes archiving 2.37 GB in three days for
    4.88 MB of live-file gain, 93 of 99 consecutive passes entered after less
    than 1 MB of growth. **Disclosed cost:** the stamp names the PRE-pass
    size, so after a high-gain fold the next pass waits until the file
    outgrows what the last one read — deliberately conservative, and the 20 MB
    tripwire below is what watches it. No floor is claimed when the ledger
    states none: no stamp (nothing has folded here yet), a leading row that
    cannot be read at all (the caller's own read reports that corruption — the
    guard neither raises it early nor stands in for the pass), or a recorded
    size that is not a positive count.
  - **The per-process memo remains, for the pass that changed nothing.** It
    holds the last attempted (inode, size) and arms after ANY pass —
    unprofitable (nothing foldable / no shrink / verify-abort) or committed,
    where it takes the COMPACTED size and the new inode. An abort leaves the
    bytes untouched, so the ledger's own floor still names the window that let
    that pass in; only the memo can throttle the retry. It is also the whole
    guard on a ledger that states no floor.
  - **No fresh foldable attempt, no opportunistic pass.** The generation-bound
    writer view indexes final attempt eligibility and its age horizon using the
    compactor's predicate. Superseded entries are discarded incrementally. A
    baseline-only residue with young/retained attempts cannot launch repeated
    full `no byte gain` passes in each worker. Crossing the horizon admits a
    pass without rescanning history; explicit maintenance still runs the full
    compactor. Committed swaps still force outside-lock writer re-preparation.
- `USAGE_LEDGER_WARN_BYTES` (20 MB) stays as the regression tripwire above the
  mechanism, exactly like the rotation-bounded log warns: it fires if
  compaction is broken, if the unfoldable residue itself reaches 20 MB, or if
  the floor above is still holding a ledger that has not outgrown what its
  last pass read. The startup note on `state/usage_attempts.jsonl`
  (`agent_startup_checks._hot_store_thresholds`) names all three, because the
  last one is declined before the pass and so leaves no typed event.

## 10. History readers: model-send reconciliation and audits

The reverse reconciliation in
[Model-send observability](MODEL_SEND_OBSERVABILITY.md#33-reverse-direction-audit-model_send-only)
(`ouroboros/model_send_seal.py`) joins model-send seals to usage-accounting
attempts. A folded attempt is absent from the live replay, so the sweep uses
the archive-aware join:

- `usage_compaction.archived_attempt_ids(root)` — the `attempt_id` set of
  every archived segment, walked through the tamper-evident header chain
  (live header → segment; segment's own embedded header → older segment, …).
  Each hop is verified, not trusted:
  - the reference is bounded twice — by the substrate's textual rule (§6) and
    by the RESOLVED path having the archive directory as its parent, so a
    symlink planted in the archive cannot import a foreign file's ids;
  - the segment must match the naming header's `source_sha256`,
    `source_size_bytes` and `source_row_count`, and must itself **validate as
    a ledger** (`_validate_records`: dense seq, legal transitions,
    well-formed rows). A tamperer who also repairs the hash still has to
    produce a structurally valid former generation;
  - **the chain's epochs must step down by exactly one and end at epoch 1
    with a header-less segment.** The source of epoch N is exactly the file
    epoch N-1 produced, so this holds by construction — and it is what makes
    re-pointing a live header at an older *genuine* segment (correct hash,
    correct counts) corruption rather than a legitimately shorter history;
  - **the archive anchors the live file.** The step-down rule alone only
    proves the chain BELOW the header, and `compaction_epoch` is as mutable as
    the rest of the row, so a forgery that repoints AND lowers the epoch walks
    a valid, short chain. The generations such a forgery orphans are still on
    disk: no segment may carry a generation newer than the live file's own,
    read from each segment's own embedded header rather than from its name —
    a live file with NO stamp being the same question with the floor at zero,
    so a stamp that was REMOVED is a corruption verdict rather than an archive
    nobody consulted. The one legal newer case is an uncommitted orphan of the
    CURRENT generation (a lost snapshot race, a crash before the swap), and it
    is recognised by what it IS: the pass writes that segment BEFORE the swap,
    so it is the byte-for-byte copy of the live file at that instant, and the
    live file only grows behind it — the orphan's bytes are still a PREFIX of
    it, and every id it holds is live, read from the descriptor the entry was
    classified through (one open per entry, never a re-open of the name, which
    could name a different file by then). Matching only its leading row against
    the live header was not that proof: the newest segment IS the previous
    generation's whole file, so a live file restored from a backup taken just
    after that compaction matched the row while being a strict SUBSET of what
    the segment held, and the attempts that compaction folded — which exist
    nowhere else — were reported absent by the join. Silent absence is the one
    verdict this surface may never reach, so the prefix is the test and a
    restored generation is corruption. The anchor scan runs in the directory the chain
    was walked in — on POSIX through the `O_DIRECTORY|O_NOFOLLOW` dir-fd
    held from the moment the handles are opened, after the live header is
    read, for the rest of the question, entries opened relative to it — so a
    directory swapped after the walk cannot hide a generation from it (a
    directory swapped BEFORE the handles open is the same power as deleting
    the newer segments: disclosed, out of the anchor's reach), and an entry
    the scan cannot list, open or read is `UsageLedgerCorrupt`: the scan did
    not complete, so the question is UNKNOWN — the data root's own handle
    included, since a bare `OSError` from THAT one open (an unreadable root,
    fd exhaustion) would escape the sweep's UNKNOWN mapping entirely. An entry
    that OPENS but is not a regular file — a stray `backup/` directory, a
    FIFO, a device — is no segment: segments are regular files by
    construction, no generation lives there, and it is skipped; one the
    kernel refuses to open at all (a UNIX socket, `ENXIO`) is corruption like
    any other unopenable entry — on the dir-fd shape; without a held handle
    (Windows, no `O_DIRECTORY`) the `stat` before the open classifies a socket
    as not regular and skips it. Where a dir-fd is held that classification is
    an `fstat` after an `O_NONBLOCK` open; where none is (Windows, or any os
    without `O_DIRECTORY`) it is a `stat` BEFORE the open, because there the
    open is the step a directory refuses and a writer-less FIFO blocks on. A
    first row that reads but does not parse is a torn segment from a crashed
    write: no evidence of any generation, left to the walk. Every path
    inspection the reader makes is typed the same way: `pathlib`
    re-raises every `OSError` but `ENOENT`/`ENOTDIR`/`EBADF`/`ELOOP` from
    `is_symlink`/`is_dir`, so the symlink bounds on both archive levels and
    on the named segment (an `archive/usage_ledger` readable but not
    searchable — the shape a `chmod -R 600 data/` hardening produces —
    refused the segment's own `lstat`) raise `UsageLedgerCorrupt`, never a
    bare `OSError`; and on a STAMP-LESS live file the question ends early
    only on the kernel's exact "no archive directory" (`ENOENT`) — a regular
    file standing where the directory belongs, or an archive that cannot be
    inspected, is UNKNOWN (typed), never a silent empty answer — and every
    case that reads anything then goes through the same typed root open.
    Disclosed, stamp-less anchor: with the floor at epoch zero every parsable
    regular file in `archive/usage_ledger/` that is not a byte-prefix of a
    stamp-less live file is `generation newer` — a fresh ledger started
    beside a surviving archive (a reset that deleted the ledger alone) leaves
    every history question a `generation newer` corruption verdict until the
    fresh ledger has compacted as many times as the old one had — after which
    the surviving segments fall under the epoch floor and are silently ignored,
    their ids absent from every history question — and an operator's stray `notes.jsonl` there is skipped on a
    stamped ledger but is corruption on a stamp-less one; move or delete the
    archive with the ledger, or keep both. Disclosed, lock-free readers: a
    compaction that commits between a question's live-header read and its
    anchor scan makes that ONE question UNKNOWN (`generation newer`: the
    freshly committed segment is prefix-tested against the new live file);
    the next question walks the new chain;
  - the archive directory must be **this data root's own**: neither
    `archive/` nor `archive/usage_ledger` may be a symlink, the resolved
    directory must be exactly the resolved root's archive path, and no segment
    may be a link. Otherwise "the segment resolves next to the archive
    directory" is a tautology — both sides resolve through the same link. On
    POSIX the reader enforces this **at the open**: the segment is opened
    `O_NOFOLLOW` through the same `O_DIRECTORY|O_NOFOLLOW` `dir_fd` handles
    the writer uses — opened once, after the live header read, and held for
    the rest of the question, chain walk and epoch anchor alike — and is
    fingerprinted and read from that fd — so a link planted after any
    path-based look is an open error, never a read through it (a
    byte-identical copy behind a link hashes perfectly; only refusing the
    traversal itself defends); a named segment that opens but is not a
    regular file, or whose read fails, is typed `UsageLedgerCorrupt`, never a
    bare `OSError` the sweep's UNKNOWN mapping would miss. Windows keeps the
    path-based checks, best effort, by the platform predicate;
  - per-segment results are cached by path, but the hit requires the file's
    fingerprint (inode/device/size/mtime_ns) to match, so a segment deleted
    or rewritten after a warm read surfaces as `UsageLedgerCorrupt` instead
    of keeping an audit's "logged" verdict alive. A fingerprint is not proof
    of identity, though: an in-place same-size rewrite within the filesystem's
    timestamp granularity keeps it. So a hit ALSO requires a file whose mtime
    has settled (> 2 s old) and an entry younger than 60 s — a cached read is
    evidence with a shelf life. **Residual, disclosed:** a rewrite that
    restores `mtime_ns` exactly can still be answered from a warm entry for up
    to that minute. Closing it would mean re-hashing every segment on every
    question, which is the quadratic cost this cache exists to avoid; the
    chain hash still binds every segment the answer depends on, and any
    process that starts, or asks a minute later, re-hashes;
  - the union over a whole chain is cached by the chain's identity
    ((`archive_rel`, sha) per hop). A bulk seal audit selects its manifests
    before reading live attempts, then lazily obtains ONE validated archived-id
    set for that pass. It retains an UNKNOWN archive answer for the rest of the
    reverse pass without suppressing independent forward checks. The next pass
    reads anew with the existing fingerprint, hash and TTL checks; TTL expiry
    inside a batch does not multiply the history read by its seal count.
    Honest about the rest of each archive query's cost: the epoch
    anchor runs on every question, ahead of that cache — one directory listing
    plus a bounded first-row read per entry the chain did not walk, and a full
    read only of an entry that claims a newer generation. The union cache is
    bounded (the chain key changes at every compaction and only the newest
    chain can be asked again), so a long-lived process does not keep one
    archived-id set per epoch it has ever seen.
- `usage_compaction.usage_attempt_recorded(root, attempt_id, live_ids)` —
  membership in live replay ∪ archive.
- A live leading row that cannot be **read at all** (bad UTF-8, bad JSON, not
  an object) raises `UsageLedgerCorrupt` rather than returning "no baseline
  header". `None` from `_live_baseline_header` means a readable first row
  that is not a stamp — which is either "no compaction has happened" or "one
  has and its stamp is gone". That function cannot tell them apart and no
  longer claims to: the archive does, because the epoch anchor runs on a
  stamp-less file too, and the wrong answer there turns a folded attempt into
  a reported orphan seal.

The model-send reconciliation contract is that the
reverse sweep's "no attempt row" verdict (`orphan_seal`) must consult this
union, not the live replay alone; an unreadable/mismatched segment is the
sweep's existing UNKNOWN → skip-pass case (fail-soft, the API raises typed
`UsageLedgerCorrupt`). The baseline header in the live file is the structural
signal that the live replay is not the full per-attempt history.

`legacy-import` needs no such lookup: its rows are never folded (§3), so its
existing live-replay dedup keeps working under a lost watermark.

## 11. Module placement

`ouroboros/usage_compaction.py` (domain D16) owns fold policy +
archive/verify/swap + history readers. It imports FROM `usage_ledger`
(substrate) and `_usage_rows` (aggregation leaf); `usage_accounting` calls
INTO it from `reserve_attempt`. The substrate stays policy-free (it learns
only the baseline row kinds' validation), the one-way seam
`usage_ledger ← usage_accounting` is unchanged, and the compactor — which must
know the aggregation semantics — lives beside the aggregation, not inside the
byte authority.

## 12. Invariants (pinned by tests/test_usage_compaction.py — the pass side,
invariants 1-4, 6, 7, 9, 10 — and tests/test_usage_compaction_archive.py —
the archive reader, invariants 5 and 8; shared fixtures in
tests/fixtures_usage_compaction.py)

1. **Byte-exact money**: decimal sums of `cost_usd` /
   `reservation_upper_bound_usd` over finals are identical before/after, and
   the NON-money projection of `usage_projection` (global + per-root incl.
   limits) and of `usage_breakdown` (all axes) renders equal dicts: state
   counts and folded weights, physical calls, token sums, finality,
   subscription sessions, per-root limits, every axis shape. Exact cash is
   checked independently of six-place rendered values. The historical
   `2.4675885` float-ordering example now renders `2.467588` on both sides;
   tests deliberately pin the corrected half-even result and unchanged exact
   charges. A sum needing more than ambient 28 digits (10²⁸ + 1) retains its
   last digit, checked with an independent wider-context oracle.
2. **Correctable attempts never fold**: reserved/dispatched/unresolved and
   administratively abandoned chains survive verbatim (modulo seq). The exact
   attempt accepts its valid terminal transition after compaction, while an actual
   late settlement/release becomes foldable under the ordinary rules. Historical
   baseline groups remain aggregates (`tests/test_usage_abandoned_ledger.py`).
3. **Crash-safety**: a failure injected at the ledger rename ITSELF leaves a
   byte-identical, valid, further-usable ledger, with the archive segment
   already on disk holding the exact source bytes; the archive directory
   chain is fsync'd before the swap — including on the RETRY after a pass that
   died on that fsync — and a POSIX directory-fsync failure aborts with the
   ledger untouched. The live file's own half is pinned the same way: the
   candidate temp is fsync'd BEFORE the rename (without it the renamed inode
   can hold unwritten data — neither the old ledger nor the approved new one)
   and the ledger's directory after it.
4. **Budget limits are preserved**: root/global enforcement thresholds are
   unchanged across compaction.
5. **Model-send join survives**: every pre-compaction `attempt_id` remains
   resolvable through live ∪ archive, across chained compactions; a tampered
   segment, a re-hashed but structurally broken segment, a deleted segment
   behind a warm cache, a same-size rewrite once the cache window closes, an
   out-of-archive reference, a symlinked archive directory, an epoch-skipping
   chain, a rollback the archive out-anchors — a previous generation restored
   over the live file, whether or not the stamp came with it — a directory
   swapped between the chain walk and the anchor scan (a look-alike carrying
   the newest NAME with the forged live header included), an archive entry the
   anchor cannot open (the data root's own handle included), a named segment
   that is not a regular file, an unreadable leading row, and a path
   inspection the kernel refuses (the symlink bound on either archive level
   or on the named segment; the stamp-less archive check, which ends the
   question early only on an exact `ENOENT`) are each typed corruption
   (UNKNOWN/skip), never silent absence and never a bare `OSError`; an
   uncommitted orphan segment of the live generation is NOT corruption (its
   bytes are still a prefix of the live file), a stray directory or FIFO in
   the archive — an entry that opens but is not a regular file — is no
   segment and is skipped on both shapes of the scan (the FIFO cannot hang
   the question; an entry the kernel refuses to open at all, a UNIX socket,
   is corruption on the dir-fd shape and skipped by the path shape's
   stat-before-open), and the chain union is built once per chain and cached
   within a bound.
6. **Idempotency survives**: subscription/external replays after compaction
   dedup (no double charge) and still conflict-check; legacy import stays
   correct with and without its watermark.
7. **Trigger policy**: no compaction below threshold; the growth guard holds
   ACROSS processes — the floor a pass stamps into the ledger throttles every
   other process and every freshly started one, and the per-process memo still
   throttles a pass that aborted, a ledger that states no floor, and one whose
   leading row cannot be read (which stays the normal read's error to report);
   verify-abort leaves the ledger untouched; the pass is entered with the
   monetary lock demonstrably HELD *and* with that lock's heartbeat wired
   through — the parameter is required, so a caller that drops it raises
   instead of proving nothing.
8. **Structure**: baseline rows only at head — a header is refused for its
   POSITION while the identical block validates at the head, and a group row
   cannot rejoin a closed block; header counts must close and agree with the
   block; `pre_compaction_seq` is unique, increasing, and legal only under a
   stamp; quarantine/`integrity_degraded` semantics unchanged.
9. **No pass loses a concurrent charge**: a row appended between snapshot and
   swap aborts the pass and survives byte-for-byte, with money equal to
   before-plus-that-row — including a row that lands between the pre-swap
   re-check and the rename, refused by the re-proof inside the atomic writer
   at the last instant; the lock's tier is chosen by the capability
   predicate — on the enforced tier it is kernel-held (flock / LockFileEx on
   the lock fd) and a kernel refusal that is not contention fails the
   acquisition closed rather than degrading to the name protocol, on the
   name tier the pass refuses to run while appends continue (a typed
   `usage_ledger_compaction_refused` event, once per process per data root)
   — owner-aware,
   and heartbeaten through the long span, with ownership proven immediately
   before each snapshot re-check — including inside the atomic writer, once
   the temp is durable and right before EVERY rename attempt, retries
   included, ownership first (a loss as the commit section is entered aborts
   before the pre-archive look and writes no orphan; a loss at the archive
   aborts before the re-check is even asked; a loss while the temp is
   written, or between two rename attempts, refuses the replace — as does a
   charge that lands between them, and a hold lost the instant the in-swap
   look answered True refuses it too, so the last proof precedes the rename
   by one syscall); a pass that loses the hold abandons its work at its next
   proof instead of swapping (bounded, not absolute: a charge landed by an
   out-of-protocol holder inside that one syscall is erased by the rename —
   on POSIX its bytes are read back from the old inode held open across the
   swap, quarantined with `integrity_degraded` raised, and the pass raises
   typed instead of returning a receipt; on Windows the loss is silent,
   disclosed); of two reclaimers
   racing over a stale lock at most one can evict (eviction only under the
   held flock on the judged fd); a creator evicted while still lock-less
   never returns a descriptor, nor does one whose own identity cannot be read
   — and that one leaves no live-pid stamp behind either; a heartbeat after an
   atomic replacement of the lock file answers False; a writer that cannot take the lock refuses in
   typed form rather than appending without it; and the swap reports a
   receipt only for bytes it re-read at the path.
10. **Provenance is bounded**: `pre_compaction_seq` names a row inside the
    header's declared source range; the archive directory is the data root's
    own, through no symlink — enforced on POSIX at the open/create itself via
    `O_NOFOLLOW` `dir_fd` handles for both writer and reader, so a link
    planted after any check can neither receive nor serve history; the epoch
    anchor is scanned through the very handle the chain was walked in,
    entries opened relative to it, so a directory swapped in between cannot
    hide a generation; and no archived segment carries a generation newer
    than the live file's own (bar an uncommitted orphan of it, proven by
    still being a prefix of that file), whether or not the live file carries
    a stamp.

## Writer continuity and qualification

`usage_ledger.py` owns one cross-process lock for validation/reservation/transition/append/fsync; accounting imports it, never vice versa. Warm writers borrow one generation-bound `_usage_rows_memo` view: only unseen/appended rows and touched IDs/roots/billing groups update validation (including late-receipt rights), final IDs and exact cash. Cold/replaced views parse a captured newline-aligned extent outside the lock, then prove file identity and cache generation and reconcile the suffix inside; replacement, including committed compaction, releases and reprepares. Strict projection, root refresh, review-wave, seal audit and custody maintenance readers reuse this SAME prepared source: they capture private immutable row references and a scalar generation under lock, then copy/render outside it. `read_usage_records(final_only=...)` supplies detached nested snapshots; full-record import remains atomic under its original lock. Seal manifest selection precedes the snapshot; maintenance retains sequence rechecks and review/late-owner policies. Display memo/render generations never retain mutable writer indexes or resume-state maps. Public full-record/resume readers retain snapshots. Cache publication follows durable append; append/fsync uncertainty invalidates it. Resume trusts the validated prefix without rereading it: within one inode, rows are only appended under the lock (locked quarantine truncates only a rejected tail), and history changes only by atomic replacement. Its new inode, like a shrink or a same-size rewrite with a new `mtime_ns`, forces preparation and full validation. A same-inode rewrite that grows the file (an editor saving in place, say) is outside this protocol: the warm view parses only the bytes past its offset, which may happen to be valid, so rewritten history can stay unseen until the next replacement or cold preparation. Rehashing history on each warm write would restore the per-append full-history cost this view removes. Full, incremental and cold preparation share `_decode_record`: UTF-8 JSON objects, with only real empty line separators ignored (whitespace-only or alternate-encoding records are invalid). Exceptional tail quarantine remains locked and loud; middle corruption fails closed. Display `allow_stale` never authorizes admission. Whole-work group keys share one pure legacy-root fallback rule in replay and the writer; root identities remain distinct. Compaction preserves the group cap source/revision and explicit unlimited `None`, checks group projection equality, and independently compares exact raw cash/bounds by root and group as well as globally.

`_usage_money.py` shares reversible precision-60 Decimal cash contributions with full replay and compaction, trapping unintended rounding. Baseline strings/raw monetary number literals retain precision. `monetary_scope_key` defines root selection once for replay and the incremental index; absent/None/empty roots retain their existing meaning, and future metadata gains no invented authority. Admission keeps six-place half-even bucket rounding and the existing `1e-9` allowance; the historical float ordering artifact at exact `2.4675885` now consistently renders `2.467588`. Historical charges are not rewritten. Rich subscription/window/non-money projections retain full replay. The same writer view indexes fold eligibility using compactor policy and timestamps: baseline-only residue cannot repeatedly launch an unprofitable full pass, while eligible attempts still use the existing locked durable compaction.

`_usage_wait.py` retries only positive pre-send acquisition contention on the same stack. Typed platform outcomes distinguish contention (including observed O_EXCL name existence followed by ENOENT at the probe) from kernel/name-tier refusal, unreadable identity, permission failure and unknown. Managed operations check existing cancellation, deadline, lifetime, budget/fence controls between short slices; ordinary chat and Presence use the existing interactive idle wait window, narrowed by any explicit deadline. Time keeps running. Each acquisition has a stable episode ID with entered/ended checkpoints; the existing event and progress histories retain both episodes across Chat/Logs reload without budget-pause semantics or cancellation of supervising external runs. Reservation admission rechecks fresh caps; a reserved-unsent wait preserves its manifest, paid stamp and one physical claim. Proven no-send interruption returns that local claim even if bounded release fails; unresolved cleanup retains reserved capture/bound. Async mutation preserves ContextVars and joins on cancellation. Post-response settlement remains bounded: the original answer/usage/candidate returns once, an unsuccessful settlement retains dispatched/unresolved custody, and no settled-only learning or resend occurs. Cancellation after receipt joins accounting (including its unresolved fallback), adopts the final capture, and retains the exact response representation through `llm_observability.retain_cancelled_response` in the existing private call CAS before propagating cancellation. The ordinary `read_call_payload` consumer can recover it; native wire custody remains with its transport. Stop/deadline never resumes caller execution. Retention I/O failure is logged and attached to the cancellation with the original response, never reported as successful persistence. Search's physical-send fact is monotonic: an empty completed response's fallback runs outside that backend's cleanup scope. Search and review share these contracts; review rows are excluded from generic abandoned sweeping, so automatic eventual release is not promised.
### Serial scale runner

Run SERIAL, with fake providers and synthetic data only:

```sh
python -I -S scripts/safe_test.py --temp-parent /tmp -- .venv/bin/python -m tests.usage_writer_scale --writers 10 24 --seconds 60 --owned
```

The default fixture creates 92k meaningful transitions with candidate, token,
processing and root provenance (~140 MB), then actually compacts to an active
85k-row/~130 MB generation. Each spawned process performs real reserve,
dispatch, local stub send and settlement with fsync. Each cycle starts with a strict `usage_projection(root_task_id="dominant")`; observed reads/second are reported for EACH process. A separate cold audit process reads and validates every record, runs real seal reconciliation and abandoned-usage maintenance, and repeats at a five-second target cadence (actual cadence, waits and holds reported). Warm, simultaneous cold and
committed compaction/replacement phases record each writer's stage wait/hold,
preparation/parse counts, completed cycles, longest gap, RSS and exit status;
separate send-ID files detect duplicate sends. A precision-200 raw-Decimal
oracle checks quiescent cash, every final row including unknown/future metadata, root/global counts, finality, subscriptions/windows and processing facts. Every writer must progress without a 45 s gap AND keep its raw acquisition maximum below 45 s; ordinary sliced wait episodes are counted separately from actual acquisition episodes over 45 s, and any such continuity extension fails; a single legal compaction exceeding that hold fails explicitly.
`--owned` binds each writer to the existing `TaskModelWait` with native controls, a populated queue snapshot (running shared-root tasks, 32 pending tasks, one active sibling-root fence), and actual global/root caps. It reports control/fence checks inside and outside the monetary lock; the fixture grants no control bypass. Omit the flag for unowned comparison. Small `--attempts 300 --writers 2 --seconds 2` runs are smoke checks only.
All runtime/control Python sources, file modes and relevant scrubbed mode environment are pinned, including `platform_layer` and the launcher. Results and candidate hashes remain in the launcher's retained temporary tree,
never product history. Measurements must identify the source generation; old
PR measurements are not evidence for a new candidate. Do not run this workload
against live data or alongside another scale run.

For a baseline, `--worker-source` accepts a read-only base archive inside the
same launcher temporary root and `--worker-ref` labels that exact reference.
Worker/audit initialization failures publish explicit readiness failures and retained error reports rather than leaving the barrier waiting. A baseline without `_usage_wait` is labelled as lacking that instrumentation. Each worker records its imported accounting source path; the result also hashes
the base source. Fixture construction, initial compaction, the measured phase compactor and the independent oracle still belong to the current runner. The receipt explicitly marks baseline compaction qualification UNSUPPORTED; a baseline-worker run must never be labelled a fresh unmodified-baseline compaction measurement. Report this distinction and expected baseline failures.
