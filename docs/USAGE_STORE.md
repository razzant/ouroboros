# Usage store: the monetary authority

`state/usage.sqlite` is the one monetary authority (store module
`ouroboros/usage_store.py`, policy owner `ouroboros/usage_accounting.py`). It
keeps ONE row per physical attempt and, beside it, every current fact the
ordinary readers ask, maintained in the transaction that changes it. No
ordinary path (admission, send, display, status, task start, maintenance
candidates) aggregates the attempts: each reads summary rows and addressed rows,
so its cost does not grow with history. The public `usage_accounting` /
`usage_admission` API (names, shapes, exceptions) is the same as in the
journal era.

## 1. Schema

The database runs a rollback journal (`journal_mode=DELETE`) with
`synchronous=FULL`.

| table | holds |
|---|---|
| `attempts` | one row per attempt id, UPDATEd on every transition; `revision` counts its transitions (the old `expected_seq` role). Typed columns for ids, state, attribution, timestamps (`ts_reserved`, `ts_dispatched`, `ts_final`, `ts_last`), money as canonical decimal text, tokens (NULL = not reported), caps, review/route attribution, local-answer owner; `weight` (1, or the folded attempt count of an imported aggregate); every other field verbatim in `extra`, so a row decodes to exactly what was written. Indexes: root, group, task, the open set (partial: reserved/dispatched/unresolved and abandoned settlements), `(category, ts_last)`, review wave, route. |
| `summaries` | `(scope, key)` → the full bucket the readers render: the cash 5-tuple as decimal text, state counts, unknown/priced/open/non-final counters, sessions and windows, processing summary, token sums with presence counts, physical calls, cache TTLs and the cap multiset; `accounted_num` (REAL) orders the costliest tasks and is never read as an amount. |
| `bindings` | the earliest root/group cap binding (`BindingIndex` semantics: carried, `legacy_live`, `legacy_default`, explicit unlimited `None`). |
| `dirty_owners` | task and root ids whose stored cost projection may be behind, with the revision that dirtied them. |
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
  conflicting one is refused.
- `mark_dispatched` re-checks known spend against the global, root and group
  limits and the owner Pause/admission fences in the same transaction.
- Recovery that raced a settlement passes `expected_revision`; the UPDATE
  matches it in its WHERE, and zero rows updated leaves the settlement
  authoritative.
- `Txn.record_recovery` updates only custody metadata and the row revision; money
  and late-receipt rights stay unchanged. Definitive terminal/gone observations
  suppress HTTP probes; retained receipts still settle.
- One-shot kinds compare identity, not payload: a subscription session's later
  model or token observation replays the stored row.
- Nonfinite money is refused before anything is written.

## 4. Reads

Money and displays read the same rows. A display read (`allow_stale`) waits
only the short display wait, then reports the fact unavailable — never zero.
`read_usage_records` (every row) serves the explicit audit and the export; a
test pins its call sites. The consciousness allowance selects its roots and
window rows through the `(category, ts_last)` and root indexes. Terminal
maintenance takes recovery candidates from the open-set index and projection
candidates from `Txn.dirty_owners()`. After an addressed projection succeeds or
is equal, `ack_dirty_owner(owner_id, revision)` removes only that revision; a
new receipt, failure, missing result or live ownership keeps the debt. A
missing result is one `stat` before the ownership read: its exact revision
stays, so a result that lands after the last receipt (child copyback, a body
restored from quarantine) is projected on a later pass without a new receipt.

## 5. Lock tiers

One configuration per installation, decided by the import with
`platform_layer.kernel_file_locks_enforced` on the lock directory, recorded in
`meta.lock_tier` and in the database header's `application_id`, so every
process chooses the protocol before it opens the file.

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

The server runs `migrate_from_journal(root)` at lifespan start on every door
(a providerless install included), before any request, worker or supervisor;
the supervisor's liveness phase `startup:usage_store` then finds it completed.
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
   copies under `archive/usage_import/`, the watermark).
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
a journal that changed after the import, or one without a store, is never
overwritten: move it aside by hand to export (the rows an older release added
after the import are not in the store).

## 8. Explicit history audit

The startup seal audit (`model_send_seal.reconcile_model_send_seals`, run in
its own interpreter) joins seals with the store and, for ids the store does
not hold (attempts folded before the store existed), with the retained
evidence: the imported journal and the archive segments, a plain id scan
without chain verification. Evidence that cannot be read is unknown, never an
orphan accusation.

## 9. Residuals

- Durability on a name-tier mount is best effort, as it was for the journal:
  the store relies on the mount's own write and rename semantics.
- Two hosts sharing one data directory are not a supported configuration (the
  name lock is per host).
- Result-only edits do not dirty money: ordinary receipts maintain projection debt;
  a result-only accounting repair belongs to explicit repair/import.
- An owner whose result never appears stays in `dirty_owners` and costs one
  `stat` per maintenance pass; absence is never acknowledged, so only a
  projection (or that explicit repair) retires it.
