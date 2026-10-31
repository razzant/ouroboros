# Usage store: the monetary authority

`state/usage.sqlite` owns money (`usage_store.py`; policy: `usage_accounting.py`).
Each physical attempt's transaction maintains its row and summaries.
Ordinary reads use current/addressed facts, never history. Custody owns
execution coverage, not amounts. The public `usage_accounting` / `usage_admission`
names, monetary shapes and exceptions retain the journal-era contract; scoped
coverage is an opt-in projection. Storage rules import no pricing/budget policy:
changing that policy cannot redefine valid storage, and locking or repair cannot
change what an attempt costs.

## 1. Schema

The shared store core defaults to `journal_mode=PERSIST`, `synchronous=FULL`
and a 64 MiB idle journal size limit. A retained rollback journal avoids file
creation/deletion on every commit; DELETE-mode older writers can interleave
only in turns: on Windows a live PERSIST connection keeps the journal open, so
a DELETE-mode commit beside it fails (SQLITE_IOERR_DELETE) and the next reader
rolls it back. Every connection that may write opens through `store.connect`;
the `query_data` SQL child's read-only reader never commits.
WAL keeps FULL and requires all three facts: SQLite >= 3.51.3 (or the 3.50
branch >= 3.50.7 / 3.44 branch >= 3.44.6, which carry the WAL-reset fix: a race
between two connections, in threads or processes, writing or checkpointing the
same WAL file at once, sqlite.org/wal.html §11), enforced locks, and a proven
local filesystem. Darwin/FreeBSD use `statfs` (apfs/hfs/ufs); Linux uses mountinfo
and known local filesystem types, with every overlay backing directory
verified; Windows requires `GetDriveTypeW` = DRIVE_FIXED for the containing
volume. Unknown or remote filesystems keep PERSIST. A working lock probe alone
does not prove WAL shared-memory support. The gate is a fact about the opening
runtime, not evidence that a file was damaged.

Lifecycle owners choose the journal (`ouroboros/store_journal.py`): private
builds publish their chosen header; lifespan boot calls explicit
`initialize_journal=True` for current facts, money and dialogue before producers.
`body_sha` and the repair CLI grant no journal authority. Boot checks existing
child headers through the custody directory walk, without SQL on compatible
children or creating absent stores; incompatible WAL leaves before lazy access.
Export, switch and retier leave DELETE (`prepare_handoff`); a new tier is
published only after conversion under the former protocol.
Ordinary opens keep the WAL/rollback boundary: local PERSIST on rollback, WAL
unchanged, incompatible WAL refused typed. `write=False` is still `mode=rw` and
may write SQLite bookkeeping. Boot promotion is optional: an early lease defers
it with a warning while rollback keeps serving. Required demotion retries only
contention within one `wait_sec` budget; exhaustion is retryable `StoreBusy`,
other failures propagate. Opens keep refusing incompatible WAL until a lifecycle
owner succeeds; no failure authorizes unsafe WAL continuation.

| table | holds |
|---|---|
| `attempts` | one row per attempt id, UPDATEd on every transition; `revision` counts its transitions (the old `expected_seq` role). Typed columns for ids, state, attribution, timestamps (`ts_reserved`, `ts_dispatched`, `ts_final`, `ts_last`), money as canonical decimal text, tokens (NULL = not reported), caps, review/route attribution, local-answer owner; `weight` (1, or the folded attempt count of an imported aggregate); every other field verbatim in `extra`, so a row decodes to exactly what was written. Indexes: root, `(root, ts_last)`, group, task, the open set (partial: reserved/dispatched/unresolved and abandoned settlements), `(category, ts_last)`, review wave, route. |
| `summaries` | `(scope, key)` → the full bucket the readers render: the cash 5-tuple as decimal text, state counts, unknown/priced/open/non-final counters, sessions and windows, processing summary, token sums with presence counts, physical calls, cache TTLs and the cap multiset; `accounted_num` (REAL) orders the costliest tasks and is never read as an amount. |
| `bindings` | the earliest root/group cap binding (`BindingIndex` semantics: carried, `legacy_live`, `legacy_default`, explicit unlimited `None`). |
| `dirty_owners` | task/root projection debt, indexed by its dirty revision; one entry per owner, removed with acknowledged debt. |
| `one_shots` | the identity of single-row kinds (external dispatch, subscription session, legacy rows). |
| `meta` | `schema_version`, `lock_tier`, the import provenance and the publication marker. |

Scopes: `global`; `root` and `group` (with the cap multiset); `task`; `kind`;
`model`/`provider`/`category` with `unattributed` for empty values (legacy
metadata/delta rows always unattributed); inside an address `root_<axis>`,
`root_task`, `root_delegated` and `task_<axis>`, `task_root`, `task_delegated`
(key `"<id>|<value>"`). A breakdown of one root or task is assembled from those
rows; a pair of addresses with no scope of its own aggregates only that root's
attempts.

## 2. One reducer

`_usage_rows.summary_delta(row)` is the contribution of ONE current row to
every counter of a bucket, built from the same rules `_summary` and
`_breakdown_bucket` always used (`cash_contribution`, the per-state branches,
`_physical_call_count`, `_processing_summary`). A transition subtracts the old
row's delta from the old keys and adds the new row's delta to the new keys, in
Python `Decimal` arithmetic on the decimal text; money is exact, and six-place
rounding happens only when a bucket is rendered or admission compares (the
`_usage_money` rule, unchanged). Counts and presence multiply by `weight`;
cash and tokens of an aggregate are already sums. The cap multiset adds and
removes the row's cap literal (the bucket's cap is the minimum present);
subscription windows keep the maximum reset per route. The property tests
compare every stored bucket with a frozen copy of the journal-era list fold.

## 3. Writes

Each public call is one short transaction (`BEGIN IMMEDIATE`): read the
addressed row and the summary rows it needs, check, write the row, apply the
deltas, upsert `dirty_owners` for the task and the root, write the binding of
a first row (`INSERT OR IGNORE`), advance the marker, COMMIT. Nothing
network-bound runs inside it.

- Transition table (`usage_ledger.validate_transition`, shared with the
  import): `reserved → dispatched | released`; `dispatched → settled |
  unresolved`, or `released` only with a `before_dispatch_failed:` reason;
  `unresolved` and an abandoned settlement accept ONE late receipt
  (`settle_reason="late_receipt"`) or a typed never-started release. Ordinary
  terminal rows are immutable; an identical repeat returns the stored row, a
  conflicting one is refused. A bound exact-attempt provider-price fact also
  permits successful `settled` nonfinal → final refinement; it cannot release
  that attempt or replace an already final price. This exception leaves the
  open-set/continuation ownership predicate unchanged.
- `mark_dispatched` re-checks known spend against the global, root and group
  limits and the owner Pause/admission fences in the same transaction.
- Recovery that raced a settlement passes `expected_revision`; the UPDATE
  matches it in its WHERE, and zero rows updated leaves the settlement
  authoritative.
- `Txn.record_recovery` updates only custody metadata and the row revision; money
  and late-receipt rights stay unchanged. The abandoned-usage duty selects (through
  `Txn.attempts`) only open rows it can act on: no imported aggregate, review custody
  or abandoned non-Claudexor row, and no abandoned row whose terminal/gone outcome is
  recorded; such a row stays at unknown price and is not polled, while its ONE late
  receipt stays open to an explicit writer.
- `Txn.record_evidence` merges only physical-failure and opaque provider-binding
  evidence under a revision check. These additive `extra` fields require no
  column/index migration on an existing database. The original physical failure,
  HTTP-200 body errors included, is retained in private CAS before retry. Failure, administrative
  closure, answer completeness and monetary finality remain independent.
- One-shot kinds compare identity, not payload: a subscription session's later
  observation replays the stored row. New partial session receipts retain known
  cash and `cost_partial`, with no reservation bound; an exact subtotal remains
  exact while `unknown_unmetered` counts its unpriced remainder once per row.
- Nonfinite money is refused before anything is written.

## 4. Reads

`query_data` money SQL uses guarded direct `mode=ro` reads by default, so a small
aggregate does not copy the whole ledger. Explicit `snapshot=true` makes a checked
disposable copy, with no fallback after failure. One `timeout_sec` budget covers
acquisition (including cold preparation), copy, check and SQL; expiry names its
phase, while corruption/I/O and unowned interrupts remain failures. The optional
`remaining_sec` narrows nested admission waits without changing other callers.
The parent owns scratch and passes the whole request by file. It bounds startup
separately; after expiry it allows 0.5 s while a source may be held, 2 s after
release for the typed answer, then reaps noninterruptible work and removes scratch.

Money and displays share rows; `allow_stale` bounds display waits, then reports
unavailable. `read_usage_records` serves audit/export only;
`test_bulk_readers_are_called_only_by_the_audit_and_the_export` in
`tests/test_usage_store.py` pins its call sites. Consciousness selects
roots by `(category, ts_last)` and window rows by `(root, ts_last)`, never lifetime
history. Terminal maintenance reads open attempts and `Txn.dirty_owners()`.
Finality/display consumers opt into `usage_breakdown(include_custody=True)`;
amount-only admission, wave fit and pacing never read custody. The coverage owner
`delegate_custody_usage` observes addressed custody before money, then joins each
run's receipt once in the SAME money read as the amounts. Own/tree/sibling scopes
and split execution/ledger roots retain their identities. Custody roots share one
display wait allowance; the subsequent money read has its own allowance. Contention
reports unavailable at debug level. Pending/live/unrecorded or unreadable custody
keeps `cost_custody` open; cleanup holds do not. Counts stay ledger-derived; later
starts are not atomically fenced. Exact partial cash is confirmed, estimated
partial cash is estimated; both carry an unpriced remainder and no total bound.

Custody changes owe task/root `cost_revision` in the existing `delegated_runs`
set. No debt means no monetary read/write/emit. Refresh checks ownership at the
addressed root and acknowledges only the observed revision after an available
projection was published or already matches; concurrent changes and failed or
unavailable reads keep debt. An absent optional carrier alone emits nothing.
Missing/nonterminal owners await their terminal write without repeated projection
attempts. Fold/import/rebuild derive debt only for still-open monetary custody,
never a burst over closed history; no new poller or monetary backfill.
Money-only passes select dirty revisions newer than the saved marker plus present
owners. A successful/equal projection calls `ack_dirty_owner(owner_id, revision)`;
newer receipts, failures, missing results and live ownership keep debt. One batched
presence query precedes ownership reads. Absence retains the exact revision, so
late copyback or quarantine restore projects without another receipt.
`review_price.py` prices one review by one reviewer row from the same rows:
`review_seat_costs` groups the review category's attempts by
`(review_wave_id, review_slot_id)` through the `(category, ts_last)` index,
summing every physical attempt of a seat in one wave; the median of a row's
recent waves on its current model is shown with its count and range, no
history reads `cost unknown`, and one process keeps the table for half an hour
so `## Review` never waits on the ledger.

From money to the round and its cause: a task result's `[SUBTASK_OUTCOME]`
carries its `swarm_receipt`, whose `cost.attempts[]` names every physical
attempt of the tree by `attempt_id` (one released before dispatch has no
`accounted_upper_bound_usd`) and whose `supervisor_context_pressure[]` gives
the task's own attempts their `physical_context.round_id`. The `attempts` row
(`query_data(op="sql", db="money")`) is the authority: its state, and in
`extra` the release `reason`, `physical_context.round_id`, the sent request's
`candidate_manifest_ref` and `physical_failure.evidence_ref`, the complete
failure source in the observability CAS (`query_data(op="blob", sha256=...)`).
The worker's `llm_api_error` row names the same attempts
(`physical_attempt_id`, `ledger_attempt_ids`) with the classification the loop
acted on; a task with its own drive writes it there and in the installation's
`logs/events.jsonl`, so it outlives the drive that retention or the owner's
storage run deletes (`docs/PERSISTENCE.md`), as the attempt row and its blob do.

## 5. Lock tiers

One configuration per installation, decided by the import with
`platform_layer.kernel_file_locks_enforced` on the lock directory, recorded in
`meta.lock_tier` and in the database header's `application_id`, so every
process chooses the protocol before it opens the file. On POSIX closing any
descriptor of a SQLite file releases every lock the process holds on it, an
open transaction's on another thread included. So `store.connect` counts its
connections per inode, and every raw read in a process that may hold a store
(the header probe, the Files panel, the read tools, artifact capture:
`store.open_for_read`, `iter_file_bytes`, `read_file_bytes`, `copy_file_bytes`)
borrows one shared descriptor per inode that closes only when no counted
connection has the file open, as SQLite's unix VFS defers its own closes. Under
WAL every open connection also holds locks on the `-shm` index, so these readers
refuse a WAL index, told on every OS by its name and identity (the regular file
the volume resolves a SQLite database's name plus `-shm` to), before opening it
(`store.is_wal_index`, `WalIndexRefused`; the Files panel answers 403); the folder
copies, captures, delegated snapshots and artifact listings built on them leave
it out as an excluded member, tracked or not, never read and with no content
baseline (a copy rebuilds its own from the copied `-wal`, which carries no lock),
and search skips SQLite's journal and WAL files. The rule assumes this process's
own databases never end in `-shm`. Under WAL a byte copy of a live store (a
download, a file or folder copy) can hold an older state (newer commits stay in
its `-wal`), be torn by a checkpoint running meanwhile, or fail to open when a
folder copy pairs the file with a `-wal` taken at another moment; an exact copy
comes from a stopped Ouroboros, whose last close checkpoints the log into the
file. The connection, tier,
file-identity and transaction core is `ouroboros/store.py`, shared with the
current-facts store; `usage_store` keeps the money error vocabulary over it.

Ordinary transactions lease one lazy connection per process, file and thread.
Durability pragmas run at connection open; acquisition and commit waits retain
their existing per-operation budgets. Returning a lease leaves no transaction
open. Thread exit, fork, file replacement, export and a folder's move
(`close_connections_under`, each process its own) release cached handles;
the raw descriptor follows the last connection. Fork callers must be outside
transactions. Export/retire closes all idle handles and leaves DELETE mode
after checkpointing, before a rename; an active lease refuses handoff with the
usual busy outcome. Any switch that retains a store while changing to an older
SQLite runtime must also call `store.prepare_handoff` after quiescing writers.
This includes switches whose target still contains the store module.
Private builds checkpoint and close their sole connection before publication:
a WAL build keeps its header and leaves no log or index under the temporary
name, a rollback build no journal. A refused switch (`forget_handoffs`) keeps its
stores in the rollback journal until the next boot chooses again.
Caller-owned raw connections must also close before rename on every platform.
At normal interpreter exit the pool closes before Python clears the inode registry
and its locks; thread exit closes that thread's connections earlier.

- `enforced`: SQLite's own file locks. `usage_store.hold` is the acquisition
  primitive the existing waits slice (`_usage_wait`): sends retry it in short
  slices that honour Stop and deadlines; maintenance waits the money-lock
  timeout; a COMMIT waits for readers to drain.
- `name` (the probe's name tier: Drive, FUSE, NFS without kernel locks):
  SQLite opens with `unix-none`/`win32-none` (no locking of its own) and EVERY
  access, reads included, runs inside the existing name-protocol money lock
  (`usage_ledger._named_lock`, owner-aware stale handling). On Windows a probe
  refused while a holder deletes the name, or refused persistently (an
  unreadable lock file, a restricted directory), counts as contention: an
  owned pre-send wait continues until the task's own controls end it, an
  unowned one raises after its bound.

The probe's refusals that mean the filesystem takes no kernel locks at all
are exactly `EOPNOTSUPP`/`ENOTSUP`/`ENOSYS` (on Windows `ERROR_INVALID_FUNCTION`
and `ERROR_NOT_SUPPORTED`, mapped onto them). The refusals of an enforced lock
that mean it is held by someone, so the acquirer stands down and re-contends,
are exactly `EAGAIN`/`EWOULDBLOCK` (on Windows `ERROR_LOCK_VIOLATION`); every
other refusal, `EACCES` included, fails closed. `ENOLCK` selects the name tier,
which the money lock refuses: on such an install every money access fails
typed until `state/` moves to a filesystem that locks. A store file that
cannot be read is `UsageLedgerCorrupt` for every caller and is never replaced.

## 6. One-time import

The server runs `migrate_from_journal(root, initialize_journal=True)` at lifespan
start on every door (a providerless install included), before any request,
worker or supervisor, which is also where an existing store gets this runtime's
journal (§1) and missing `(root, ts_last)`/dirty-owner revision indexes once
(failed builds keep broader scans until the next boot);
the supervisor's liveness phase `startup:usage_store` then finds it completed and
keeps the journal as it is.
A display read (`allow_stale=True`) never imports: while a journal or a
pre-ledger event chain awaits that job it reports the store unavailable, and with
neither it creates the empty store. A non-display reader that finds no store (a tool or a test without a
server) runs the same import itself; a caller arriving while it runs waits
only its own budget.

1. Decide the tier; build the schema in a sibling file.
2. Read `state/usage_attempts.jsonl` once with the validated journal reader
   (`usage_journal`): a torn final row is quarantined exactly as before, damage
   before it refuses the import. For every attempt id the LAST row is stored
   (open rows included); compacted `usage_baseline_group` rows become weighted
   aggregates; bindings are the `BindingIndex` fold of every row; summaries
   are the reducer over the imported rows; `dirty_owners` holds owners whose stored cost projection is missing or differs
   from the imported summaries; one-shot identities are stored.
3. An install whose pre-ledger import never completed imports its `llm_usage`
   events and `state.json` totals in the same job (source hashes, archived
   copies under `archive/usage_import/`, the watermark). Ambiguity is disclosed;
   no log is rewritten and no attempt invented.
4. Publish the store by atomic rename; record the header's epoch/sequence, the
   source size, hash and counts in `meta.import`; continue the marker at the
   journal's `[epoch, last seq]` so the `state.json` projection never sees a
   lower one. The journal stays in place: an older release (an in-app
   Rollback, or an automatic rollback after a failed boot of this one) still
   reads its own history up to the import.

A completed store is never imported again; later starts compare only the
journal's size with the imported one (no hashing) and report a journal an
older release appended to after a rollback as `changed_after_import`: those
rows are not merged (the export below keeps money exact across a downgrade).
An unpublished build is discarded and redone. Timing and counts go to
`logs/supervisor.jsonl` (`usage_store_migration`). The journal and
`archive/usage_ledger/` stay as evidence; nothing ordinary reads them, and
nothing writes the archive any more.

## 7. Downgrade export

A git revert alone is not a data rollback. Before checking out an older
release, stop the server and run `python scripts/export_usage_journal.py
<data_root>`: it writes `state/usage_attempts.jsonl` as a journal the older
validator accepts (a minimal legal chain per attempt, aggregates under a
`usage_baseline` header carrying the stored provenance, dense `seq`), restores
the legacy-import watermark, aligns the `state.json` marker with the journal's
own, and renames the store `usage.sqlite.exported-<UTC>`. A later upgrade
imports that journal again, including what the older release added. The
journal the import kept is set aside as `usage_attempts.jsonl.pre-export-<UTC>`;
a journal that changed after the import (its sha256 is neither the imported nor
the exported one), or one without a store, is never overwritten: move it aside
by hand to export (the rows an older release added after the import are not in
the store).

The export is one plan committed before any file moves. Under the write lock it
computes the journal bytes and commits an export mark (`meta.export`: the
journal's size, sha256 and row count, the names it will retire and set aside,
the `state.json` marker it replaces) in the store itself. While that mark
stands every write hold is refused typed (`UsageExportInProgress`; reads still
serve), so a late writer cannot land a row the journal does not carry. Only then does it set the
kept journal aside, write the new one, align the marker and rename the store.
An interrupted export is resumed from the mark (the journal bytes are
recomputed and must match the recorded digest; a partial own journal is told
from a foreign one by the mark's set-aside record, never by size; a retired
store is recognised by the digest in its own mark) or abandoned by the in-app switch
(`usage_store.abandon_export`): the store comes back first, then the journal,
then the marker, and the mark is cleared last, so a crash inside the
abandonment resumes it instead of leaving two authorities. Residual: a process
that was not holding the lock when the export began sees the store gone only
on its next access, where the import path brings the exported journal back in
— the quiescence census the switch runs first is what keeps that path cold.

## 8. Explicit history audit

The startup seal audit (`model_send_seal.reconcile_model_send_seals`, run in
its own interpreter) joins seals with the store and, for ids the store does
not hold (attempts folded before the store existed), with the retained
evidence: the imported journal and the archive segments, a plain id scan
without chain verification. Evidence that cannot be read is unknown, never an
orphan accusation.

## 9. Explicit OpenRouter price reconciliation

From a source checkout with its dependencies installed:

```bash
python scripts/reconcile_openrouter_cost.py --attempt-id ID
python scripts/reconcile_openrouter_cost.py --attempt-id ID --fetch
python scripts/reconcile_openrouter_cost.py --attempt-id ID --apply
```

Repeat `--attempt-id` to select several physical attempts; duplicate selections
are processed once. `--data-root PATH` selects an existing store, otherwise the
configured data root is used. Inspect is the default: no HTTP or monetary
mutation. `--fetch` permits one GET per selected attempt without a retained
price (5-second connect / 15-second read allowance, no redirect, retry or sleep).
`--apply` independently permits applying a retained validated receipt. Combine
both flags for an explicit fetch-and-apply. Output is JSON; exit 1 means at
least one selected attempt could not complete the requested action.

`openrouter_cost.py` owns generation metadata parsing and private receipt CAS.
New sends bind the original endpoint and credential SHA-256, then the first
observed generation ID as soon as headers or body expose it, so a later stream
failure does not lose the price lookup identity; contradictory IDs record a conflict. The explicit GET
uses the settings/route resolver and requires that exact endpoint/fingerprint,
without rotating keys. Missing legacy bindings cannot be guessed from a peer,
model, timestamp or arbitrary request ID. Custom endpoints supplied only through
an embedded client's constructor need a matching caller-supplied target to the
receipt producer; the source command resolves the configured OpenRouter route.

Received prices, zero included, outrank absent token counters and error shape.
A price is what the reply consumed, whichever wallet paid
(`_usage_response.consumed_cost_value`, shared by the ledger, the loop lane and
this receipt): a reply whose own usage says `is_byok` is true ran on the caller's
provider key, so its `upstream_inference_cost` (the provider-billed share) adds
to the gateway charge and the row carries `cost_basis` `byok_gateway_plus_upstream`;
without a valid upstream figure it is unknown, never $0. Every other reply keeps
its charge (OpenRouter repeats it in that field), and settled history is not
rewritten. `_usage_response.provider_cost_value` validates incoming cash; malformed prices
stay unknown despite available tariffs. The host-derived `cost_invalid` fact
survives settlement, loop and Safety telemetry. Valid alternate evidence wins;
absent/null prices permit nonfinal token estimates, never image text tariffs.
Generation 404, 401, 403, 429, mismatched ID and absent/invalid money are distinct
non-price outcomes. `Retry-After` is retained for an operator's next invocation;
the command never schedules another lookup. A valid retained price needs no
current key. It is read from the deterministic
`physical_<attempt>_openrouter_price` call manifest before any GET, with source
hash/identity verification. Non-price observations use the companion `lookup`
manifest; a later failed lookup cannot displace a retained price.

`openrouter_cost.apply_retained_receipt` validates the retained provider bytes
and submits a neutral exact-attempt price fact with an opaque binding. Shared
ledger/store rules compare that binding without interpreting provider IDs,
endpoints or credential grammar.
`usage_accounting.apply_provider_price_receipt` owns the revision-checked
price-only transition and returns `applied`, `duplicate`, `stale`, `conflict`
or `ineligible`. It preserves token, cache, processing, attribution and failure
facts. A different final price is a conflict, never last-write-wins. Receipt
retention precedes application; repeating `--apply` recovers an interrupted run.
Existing dirty-owner maintenance refreshes terminal task/root projections after
the ledger write, including after restart. Completed review attribution permits
price recovery; live review operations (including a review wave that settles after
the author's task ends), live owners and post-task work remain protected. Price
uncertainty grants no continuation rights. Applying a price to an unresolved attempt closes its monetary row, removing that row's model-handoff blocker; separate live ownership checks and incomplete-response facts remain. Refining an already-settled nonfinal row never expands the open-attempt set.

Standalone command custody uses the selected data root's existing current queue
snapshot, never the imported process's empty maps. Missing or stale ownership
evidence refuses task-backed mutation. The existing direct-root projection
also protects listed owners; incomplete or unreadable projections retain
custody. Absent/empty direct-root projections carry no proof that every direct
stack ended. Explicit `system:<source>` accounting
scopes from one-shot provider/capability probes are not task-backed review
owners; real review task IDs still require review-operation custody. The final
ownership recheck and monetary revision CAS cannot atomically fence a task
restart in another process; that cross-process race remains.

An exact price received before a later read, close, rejection or cancellation
error settles money without proving
that the response completed. The existing `stream_incomplete` fact keeps running
turns' shared unknown-response recovery (Presence included), revoked-control
no-resend and diagnostic response outcomes independent of monetary capture state.
The low-level opt-in transport-death repeat API and old records remain a
compatibility surface, not production Presence policy. Timeout, local-route, Stop and deadline exclusions
remain in their existing consumers. Diagnostic events carry `stream_incomplete`
beside monetary custody state; typed terminal local rejection keeps its existing
provider-error policy. Async first-generation binding runs in a joined worker;
failed binding can recover from the stream receipt using the explicit attempt
reservation after ambient accounting context is cleared.
Generation binding is a post-dispatch evidence write: synchronous binding and
its joined async worker use the existing bounded money-lock acquisition (up to
45 seconds), not the sliced pre-send wait. Stop/cancellation may wait for that
bound at this point; no new timeout or retry policy is introduced.

Ordinary sends, normalizers, startup and maintenance perform no OpenRouter
generation GET. There is no automatic historical correction or automatic
application of fetch-only previews. Explicit use on live history requires the
operator's separate authorization; installation alone changes no old money.

## 10. Residuals

- A subscription session's input axes are its harness's normalized split from
  2026-10-27 (unknown where the split is), the harness's own counters kept in
  `extra.harness_tokens`; older session rows hold those counters (Claude Code:
  the uncached rest), so token sums spanning the change mix the two. A sum no
  row reported stays null in the ledger, task totals, `/status` and Logs; the
  `/api/cost-breakdown` compatibility token fields, typed as integers, show 0.

- Durability on a name-tier mount is best effort, as it was for the journal:
  the store relies on the mount's own write and rename semantics.
- Two hosts sharing one data directory are not a supported configuration (the
  name lock is per host).
- Result-only edits do not dirty money: ordinary receipts maintain projection debt;
  a result-only accounting repair belongs to explicit repair/import.
- Absent owners keep `dirty_owners` debt. Successful absence rests until arrival,
  their own money change, repair or restart; a changed ledger epoch reselects all.
  Only a successful projection or explicit repair retires debt.
