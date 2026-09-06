# C6 review packet — monetary usage-ledger compaction (CPL4-C6, owner 1A)

Lane: `v7next_c6`, base `74a03082`. Owner sanction: batch №8 item 1A
(2026-09-01) — compaction of `state/usage_attempts.jsonl` in its own reviewed
lane (monetary authority). Design note ratified before code:
`docs/v7next/DESIGN_USAGE_COMPACTION.md` (commit `a1063124`); implementation
+ pins in the follow-up commit; this packet + ledger section close the lane.
Round 2 (§6) is the fix-round for the external adversarial wave against
`e2801c52`. Round 3 (§7) is the fix-round for the second wave, against
`830aa35a`: five findings were re-opened as still-open (1, 2, 3, 4, 6) and
all five are fixed here; four (5, 7, 8, 9) the wave confirmed closed.
Round 4 (§8) is the fix-round for the third wave, against `d7b487ab`: the
lock's exclusion becomes kernel-enforced, ownership is proven adjacent to
every irreversible decision, the swap re-proves its snapshot inside the
atomic replace, and the archive symlink bound moves from check-then-use to
the open itself (dir-fd `O_NOFOLLOW`). Round 5 (§9) is the fix-round for the
fourth wave (gpt-5.6-sol, read-only), against `13af62c5`: the lock's tier
becomes an explicit capability predicate with a fail-closed enforced tier,
the swap re-proves ownership and snapshot before EVERY rename attempt, the
epoch anchor scans through the handle the chain walk held, the two surviving
mutations are pinned, and the doc absolutes are stated per tier. Round 5.2
(§9, second block) is the fix-round for the adversarial lenses over round 5,
against `2dd3e017`: the acquisition itself is identity-checked, the
name-tier refusal becomes a durable typed event, the swap proves ownership
on both sides of its last look, the anchor scan classifies non-regular
entries and cannot hang, the anchor-swap pin covers the open-through-fd
half, LockFileEx refusals classify by their Win32 error, and the remaining
absolutes are bounded per tier.

## 1. Diff map (what to read, in review order)

| surface | change | why |
|---|---|---|
| `docs/v7next/DESIGN_USAGE_COMPACTION.md` | NEW — the ratified contract | invariants, fold scope, decimal rule, seq policy, crash order, trigger, CPL-5 join |
| `ouroboros/usage_compaction.py` | NEW leaf (D16, ~490 lines; round 5.2: `owned_and_intact` beats on both sides of its look, the typed durable `usage_ledger_compaction_refused` event, non-regular archive entries classified — `O_NONBLOCK` open, `S_ISREG` — and typed `_load_segment` failures; round 5.3: the heartbeat REQUIRED by both entry points, the orphan exemption a byte-prefix proof read from the classified descriptor, the anchor running on a stamp-less live file, the root handle typed, the name-tier mark following the landed row, the path-shape scan classifying before the open, the chain-union cache bounded — 1197 lines; round 5.4: the refusal mark following a True append only, every reader path inspection typed and the stamp-less check exact on ENOENT, the swap's old-inode witness quarantining an erased charge and raising instead of receipting, `_build_candidate`'s beat required — 1245 lines) | fold policy + prove-then-swap + archive + history readers; imports FROM `usage_ledger`/`_usage_rows`, called INTO by `usage_accounting` — the one-way substrate seam is unchanged |
| `ouroboros/usage_ledger.py` | `_validate_records` learns the two baseline kinds; round 5: `LOCK_REL` (lock-path SSOT) and the atomic writer routing its precondition through `replace_atomic` | head-only baseline block, exactly one header at seq 1, group rows joined by `baseline_id` + positive `folded_attempt_count`; a baseline row in an appended tail or after any non-baseline row = corrupt. Everything else (locking, append arithmetic, quarantine, resume fingerprints) is untouched |
| `ouroboros/_usage_rows.py` | `_summary` + `_physical_call_count` baseline-aware | header skipped; group rows: count axes × `folded_attempt_count`, sums added once. Weight-1 paths are byte-equivalent to the previous code (pure refactor for existing kinds) |
| `ouroboros/usage_accounting.py` | +5 lines in `reserve_attempt` | the opportunistic trigger under the already-held monetary lock; contained (never raises into the reservation) |
| `ouroboros/config.py` | `USAGE_LEDGER_COMPACT_BYTES` (8 MB), `USAGE_LEDGER_COMPACT_RETRY_GROWTH_BYTES` (1 MB) | trigger policy from config SSOT, no env knob |
| `ouroboros/agent_startup_checks.py`, `ouroboros/context_budget.py` | warn-text updates only; round 5.2: the tripwire names the name-tier refusal as a third cause and points at its event | the 20 MB WARN becomes the broken-compaction tripwire — and can tell the tiers apart |
| `ouroboros/domains.toml`, `docs/DOMAIN_MAP.md` | new module seated D16; graph regenerated via `check_domains.py --write` | manifest completeness gate |
| `docs/PERSISTENCE.md` | usage-ledger row rewritten (bounded by compaction); NEW `archive/usage_ledger/segment_*.jsonl` row | inventory truth; scan pin 123→124 in `tests/test_persistence_inventory.py` |
| `tests/test_gateway_abi3_removals.py` | one per-site allowlist row | `_build_candidate` writes the internal ledger-plane `cost_usd` key (same class as every existing ledger writer row there) |
| `tests/test_usage_compaction.py` | NEW pin suite (61 test items after round 5.3; the suite entered the 1001-1500 size band with a recorded rationale — it stood at 1512 lines for one commit, `79a1b9fb`, with the manifest stale, and the round-5.2 fold `6ad110e9` returned it inside the band without a new abstraction; 1492 after round 5.2 — and LEFT the band in round 5.3 at `e08a0392`: 1597 at the tip, in the ungated 1501-1600 zone, the manifest regenerated with it; the owner decision is stated in §9 "Round 5.3"; round 5.4: 66 items, back at 1597 after five new/extended pins — paid for by a `compacted` fixture folding twenty-one seed-then-compact preambles and by argument-list/data-literal reflows, no claim dropped) | see §3, §6, §7, §8 and §9 |
| `tests/test_lockfile_helpers.py` | +5 lock-ownership pins in round 3, +2 in round 4, +5 in round 5 (one Windows-only), +2 in round 5.2 (the lock-less creator; LockFileEx classification), +2 in round 5.3 (an unreadable own identity is never a hold; the design note's refusal sets) with the classification pin extended to the unsupported set, +3 in round 5.4 (a pid answering EPERM is alive and its aged lock is not reclaimed; ENOLCK keeps the enforced tier and the acquisition fails closed; two threads racing the first probe run one probe) with the probe pin's name-tier selector moved to EOPNOTSUPP and an ENOLCK clause, and the identity pin extended with the heartbeat's refresh clauses | the finding-1 fixes are platform primitives, so they are pinned where those primitives live |
| `ouroboros/platform_layer.py` | lock ownership: `_lock_identity`, inode-guarded stale eviction and release, ownership-reporting heartbeat; round 5: `kernel_file_locks_enforced` capability predicate (enforced vs name tier), fail-closed kernel refusal, LockFileEx error classification; round 5.2: the owner pid written BEFORE the kernel lock and a won lock returned only while the path still names it (an evicted creator re-contends), `_win32_lock_error` classifying ERROR_LOCK_VIOLATION alone as busy; round 5.3: a descriptor whose own identity cannot be read is never a hold (its live-pid stamp removed with it), `_WIN32_LOCK_ERRNOS` mapping ERROR_INVALID_FUNCTION / ERROR_NOT_SUPPORTED onto the unsupported set, both kill-tree sweeps on `force_kill_pid` (exactly 1500 lines); round 5.4: ENOLCK out of the unsupported set (fails closed), winerror 1 onto ENOSYS, the tier cache decided once under `_KERNEL_LOCK_TIER_LOCK`, `pid_is_alive` reading EPERM as alive with `pid_provably_gone` its one-line negation (1497 lines) | round 3, finding 1; round 5, finding 1; round 5.2, findings 1 and W; round 5.3, findings 1 and L5; round 5.4, R1 and R7 |
| `ouroboros/utils.py` | `replace_atomic(precondition=)`: the precondition is asked before EVERY attempt, the Windows sharing-violation retries included; returns False without replacing when refused | round 5, finding 2 |
| `ADOPTION_v7next.md` | CPL-4 row: C6 landed + verification hook | adoption gate |
| `docs/v7next/LEDGER_CORRECTIONS.md` | append-only C6 lane section | provenance |

## 2. Invariants to verify adversarially

1. **Money is decimal-exact.** Group sums are `Decimal`s of the exact JSON
   literals, carried as exact-decimal strings (`_number` accepts strings at
   every reader: validator, `_summary`, projections). Retained rows are
   verified Decimal-identical across re-serialization; a non-round-trippable
   foreign literal aborts the pass (never approximates).
2. **Prove-then-swap.** Commit happens only after the candidate bytes
   (a) re-validate structurally and (b) render EQUAL dicts through the
   production aggregation on every consumed surface (global summary, per-root
   summaries + min `root_limit_usd`, breakdown buckets on all five axes) and
   (c) match decimal money totals. Any inequality → abort → ledger
   byte-identical. Compaction is an optimization; it can only decline.
3. **In-flight rows never fold** (reserved/dispatched finals keep their whole
   chain, verbatim modulo seq) and their later transitions work unchanged.
4. **Idempotency-bearing kinds never fold** (subscription/external/legacy):
   their replay dedup + conflict checks read the live replay only. This is
   why their exclusion is structural, not an optimization choice.
5. **Crash-safety order**: archive segment written + fsync'd (file and every
   directory entry the chain created; POSIX failure is fatal, Windows is a
   disclosed no-op) BEFORE the atomic ledger swap, and the swap is refused if
   the live file changed since the snapshot. Crash anywhere = valid ledger
   (old or new generation); orphan segments harmless.
6. **seq policy**: dense-seq validation authority preserved by starting a
   fresh epoch; original seqs survive in the archive and as
   `pre_compaction_seq` on retained rows. Substrate append/resume arithmetic
   (`len(records)`-based) is deliberately UNCHANGED — check this holds.
7. **Concurrency**: everything under the existing monetary lock (`_locked`),
   which is owner-aware and heartbeaten so a long pass cannot be evicted by
   elapsed time; cache coherence is structural (atomic swap → new inode →
   every resume fingerprint refolds). No cooperative invalidation.
8. **CPL-5 join**: every pre-compaction `attempt_id` resolves through
   live ∪ `archived_attempt_ids` (hash-chained, tamper-evident, each segment
   bounded to the archive directory, revalidated as a ledger, cached per
   immutable segment BY FINGERPRINT, chain epochs stepping down to 1). The
   CPL-5 reverse sweep (NOT on this base — only its design note is) must
   consult this union and treat typed corruption as UNKNOWN/skip; contract
   recorded in the design note §10 and the ledger section.

## 3. Pins (tests/test_usage_compaction.py, 31 tests; round-2 pins in §6)

- exact money + whole-projection equality (incl. `skill_review_usage` waves)
- global + root budget refusal thresholds identical across compaction
- in-flight survival + post-compaction settle/release
- crash injection between archive and swap → byte-identical ledger, retry OK
- chained compactions: id resolution live ∪ archive; tampered segment raises
- subscription/external replay dedup + identity-conflict still enforced
- legacy import: rows retained; watermark-loss replay appends nothing
- trigger: config threshold gates the reserve path; growth-throttle after an
  unprofitable pass; verify-abort on a foreign non-canonical literal
- structure: baseline rows only at head (tail-smuggled row = corrupt; group
  without header = corrupt); quarantine + `integrity_degraded` on a compacted
  file unchanged; archive segment = exact source bytes, sha-pinned
- round 2 adds fifteen pins for lock ownership/heartbeat, the snapshot
  re-check, the archive directory chain and its fsync failure, header
  provenance and counts, bounded archive references, epoch-chain steps,
  segment revalidation, warm-cache integrity, typed corruption of an
  unreadable header, union caching, and decimal precision (§6)
- round 3 adds sixteen more (five of them in `tests/test_lockfile_helpers.py`)
  for lock-file ownership on eviction/release/renewal, the pass abandoning a
  lost hold, heartbeats inside the long span, writer exclusion at the swap and
  the absence of any unlocked fallback, the post-swap re-read, retry
  durability of the directory chain, the archive epoch anchor and its orphan
  tolerance, source-range provenance, the segment-cache windows, and archive
  symlink bounds (§7)
- round 4 and its verification add eight (two of them in
  `tests/test_lockfile_helpers.py`): two racing reclaimers yield at most one
  holder, a heartbeat after an atomic replacement of the lock file answers
  False, an append between the pre-swap re-check and the rename aborts
  without loss, a hold lost at the archive is seen before the snapshot
  re-check is even asked, a hold lost after the re-check aborts before the
  swap, a hold lost WHILE the candidate temp is written refuses the replace
  (the verification round's panel fix), and a link planted after the bound
  check can neither receive (writer) nor serve (reader) history (§8)
- round 5 adds ten (five of them in `tests/test_lockfile_helpers.py`, one
  Windows-only): a non-contention kernel refusal fails the acquisition
  closed; a stale lock is never evicted without the kernel hold; the name
  tier is chosen by the predicate and makes no kernel call; the capability
  probe decides once per directory and leaves no residue; LockFileEx
  contention reads as busy (Windows); the pass refuses the name tier while
  appends continue; a refused rename re-proves the hold and the snapshot
  before retrying (append / hold-lost variants); the epoch anchor scans the
  directory the chain was walked in; an entry the anchor cannot open is
  typed corruption; a hold lost before the first commit look writes no
  orphan (§9)
- round 5.2 adds two in `tests/test_lockfile_helpers.py` (a creator evicted
  while still lock-less never returns a descriptor; LockFileEx refusals
  classify by their Win32 error — runs on POSIX too) and, in this suite, one
  new pin plus five strengthened ones: a hold lost the instant the last
  snapshot look answered True refuses the rename; the after-recheck pin also
  requires that the in-swap look is never asked once the hold is gone; the
  anchor-swap pin carries the epoch-3 NAME with the forged live header and
  requires `generation newer`; the cannot-open pin also plants a directory
  and a FIFO (skipped, no hang) and requires `could not complete`; the
  warm-cache pin adds the directory-in-place-of-segment shape (`not a
  regular file`); the name-tier pin requires exactly one durable
  `usage_ledger_compaction_refused` row and the tripwire text naming the
  tier (§9, second block)
- round 5.4 adds four in `tests/test_lockfile_helpers.py` and, in this
  suite, two new pins plus three strengthened ones: a stamp-less ledger still
  inspects its archive fail-closed (a regular file where the directory
  belongs, an uninspectable archive); a path inspection the reader cannot
  make is typed corruption (the not-searchable segment directory, an
  `is_symlink` the kernel refuses); the swap-lie pin gains the `erased`
  variant (a charge landed inside the rename syscall is quarantined, flags
  integrity, and the pass raises instead of receipting); the name-tier pin
  gains the append-returned-False shape; the reserve-path pin proves the
  heartbeat it is handed is THAT lock's (aged lock renewed), not a callable
  (§9 "Round 5.4")

Mutation-probed red (not just green-once): `_summary` weight math, group-sum
rounding, folding of dispatched rows — each flips at least one pin.

## 4. Gate evidence (this host, isolated env roots)

- targeted: usage family (7 files) green; budget family (5 files) green;
  persistence inventory + domain manifest + rotation train green;
  test_usage_compaction 16/16 green
- full CI-shape non-serial battery (`-m "not serial and not integration and
  not browser and not ui_browser and not ui_browser_docker and not
  portable_detail and not skill_smoke and not size_ratchet" -n 16 --dist
  loadscope --max-worker-restart=0 --timeout=300 --timeout-method=thread`):
  EXIT=0, ~13.4k outcomes, 0 failed (first run had exactly one red —
  the ABI-3 alias sweep discovering the new `cost_usd` emission site — fixed
  by the per-site allowlist row, battery relaunched whole and green)
- serial pass: EXIT=0 (622 passed / 39 skipped); size_ratchet: 5/5, exit 0
  (PIPESTATUS-preserved); `ruff check . --select F` clean;
  `scripts/v7next_adoption.py` OK; `git diff --check` clean;
  `git rev-parse HEAD` verified after every pytest run
- scale smoke: 24,000-row / 11.9 MB synthetic ledger → 183 KB (65×),
  280 groups, 1.16 s pass; projections byte-equal; post-compaction reserve
  correctly refused over the folded money (accounted $1228 > $200 limit)

## 5. Known residuals (disclosed, not defects)

1. Subscription/external/legacy/review-attributed rows never fold → slow
   residual growth on delegation- or skill-review-heavy installs; the 20 MB
   WARN now names exactly this case. A future lane may fold them behind an
   archived-identity membership check (design note §3).
2. The in-compactor render fingerprint mirrors the COMPOSITION of
   `usage_projection`/`usage_breakdown` (using the same production `_summary`
   / `_breakdown_bucket` primitives). A future divergence in that composition
   would weaken the self-check, not correctness (worst case: a lawful pass
   aborts); the end-to-end pin compares the real projection functions.
3. Directory fsync is a disclosed no-op on Windows (POSIX guaranteed and now
   FATAL on failure, round 2 / finding 2); worst case there is a lost archive
   dir entry AND a swap in the same crash window — mitigated by archive-first
   ordering, disclosed in the design note.
4. A float-boundary rounding coincidence can make the rounded projections
   differ pre/post → the pass aborts and the ledger simply stays uncompacted
   (correctness over availability; disclosed in design note §5).
5. CPL-5's sweep is not on this base; its contract (consult live ∪ archive;
   corrupt chain = UNKNOWN/skip) is recorded in the design note §10 for the
   lane that lands it. `model_send_seal`-targeted gates therefore do not
   exist on this base to run.
6. **Orphan archive segments** (round 2, widened in round 3): a pass that
   loses the snapshot race, dies at its swap, or is abandoned by a lost lock
   can leave a written-but-never-referenced segment. It carries no money and
   no chain authority — readers start at the live header and follow only what
   it names, and the epoch anchor recognises an orphan of the live generation
   as legal because its bytes are still a PREFIX of the live file (round 5.3;
   matching its leading row alone admitted a restored generation too). Repeated lost races nevertheless accumulate disk: LOW
   availability / forensic clutter, not correctness. No GC by design (§13 of
   the note).
7. **Warm segment-cache window** (round 3): the per-segment cache hit needs a
   matching fingerprint, an mtime settled for > 2 s and an entry younger than
   60 s. An in-place same-size rewrite that ALSO restores `mtime_ns` exactly
   can therefore still be answered from a warm entry for up to a minute.
   Closing it means re-hashing every segment on every question — the
   quadratic cost the cache exists to remove — for an attacker who already
   has write access to the data root and can be caught a minute later, by any
   other process, and by the chain hash on every segment an answer depends
   on.
8. **Ownership is defended, not guaranteed — per tier, and bounded** (round
   3, stated per tier in round 5, bounded in round 5.2, identity-complete in
   round 5.3): on the enforced tier the lock primitives are ownership-exact
   and kernel-guarded (the acquisition itself included: neither an evicted,
   still lock-less creator nor a descriptor whose own identity the kernel
   cannot read ever returns a hold), and the pass heartbeats through its long
   span,
   immediately before every rename attempt and again after the in-swap
   snapshot look. No claim is made that a pass can never be robbed of the
   lock. The bounded claim: a concurrent holder can exist only after the lock
   file is removed by an actor outside the lock protocol (a hand repair, a
   foreign helper, a name-tier process of a mixed-tier install — in-protocol
   eviction is impossible under the held flock and the heartbeat-fresh
   mtime); such a robbery is caught at the next proof, and the irreducible
   residual is the interval between the final ownership proof and the rename
   syscall, in which a charge landed by that holder IS erased by the rename.
   *(Correction, round 5.4: this packet, DESIGN §8/§12.9 and the round-5.2
   ledger line said the loss was "then surfaced by the post-swap re-read or
   quarantined seq-misnumbered on the next read" — neither could ever see it:
   the re-read compares the NEW inode against the candidate and the archive
   segment is the pre-row snapshot, so the loss was SILENT and a success
   receipt was returned. Now, on POSIX, the swap holds the old inode open
   across the rename and reads back what landed beyond the proven snapshot:
   those bytes are quarantined — `state/usage_attempts.quarantine.jsonl`,
   which flips `integrity_degraded` — and the pass raises typed instead of
   receipting; the charge is not re-appended. Windows cannot hold the
   destination open through `os.replace`: silent there, disclosed. §9 "Round
   5.4", R5.)* In-protocol, no writer can append in the
   compare→replace window, because every writer of this ledger takes the same
   owner-aware lock and has no unlocked fallback. On the name tier no such
   claim is made at all: the pass does not run there (§5.10). The round-5
   sentence "it cannot finish while robbed" was an absolute the round-5.2
   probe refuted (PROBE-1: a row appended after the third look answered True
   and before `os.replace` was erased, receipt returned); corrected here.
9. **Epoch anchoring reads content, not names** (round 3, narrowed in round
   5, classified in round 5.2): a garbage REGULAR file whose first row reads
   but does not parse (a torn segment from a crashed write) is no evidence of
   any generation rather than corruption, so it cannot deny service to the
   whole history; a directory or special file is not a segment (segments are
   regular files by construction) and is skipped — a FIFO is opened
   `O_NONBLOCK`, so it cannot hang the question either *(qualified, round 5.4:
   an entry that OPENS but is not a regular file is skipped; one the kernel
   refuses to open at all — a UNIX socket, ENXIO — is corruption like any
   other unopenable entry on the dir-fd shape; the path shape's stat-before-open
   classifies a socket as not regular and skips it)*; an entry the scan
   cannot list, open or read IS corruption — the scan did not complete, the
   data root's own handle included since round 5.3, and since round 5.4 every
   path inspection the reader makes (the symlink bounds on both archive levels
   and on the named segment; the stamp-less archive check, which ends the
   question early only on an exact ENOENT) is typed the same way instead of
   escaping as a bare `OSError` or a silent empty answer. Disclosed since
   round 5.4: with a stamp-less live file every parsable regular file in the
   archive that is not its byte-prefix is `generation newer`, so a ledger
   reset beside a surviving archive is a `generation newer` corruption verdict
   on every history question until the fresh ledger's epoch passes the
   surviving segments, which are then silently ignored (reset both together)
   and an operator's stray JSON file
   there is corruption on a stamp-less ledger only; and the readers being
   lock-free, a compaction that commits between a question's live-header read
   and its anchor scan makes that ONE question UNKNOWN (`generation newer`),
   the next question walking the new chain — and the scan runs through
   the very handle the chain walk held, entries opened relative to it (and
   without a dir-fd the classification happens BEFORE the open, which is the
   step a directory refuses on Windows and a FIFO blocks on: round 5.3). The round-5 wording ("an entry the scan cannot list, open
   or read" beside "a garbage file cannot deny service") contradicted itself
   for a directory/FIFO/unopenable file and was false for the first two:
   round 5 made a stray `backup/` directory typed-corrupt for every history
   question (reproduced on `2dd3e017`; `13af62c5` answered) — an availability
   regression with no correctness gain, corrected here. Round 5.3 then replaced
   the orphan exemption itself: recognising an orphan by its leading row also
   admitted a restored previous generation, which is NOT indistinguishable and
   does hide ids — the attempts the rolled-back compaction folded exist
   nowhere else. An orphan is the pre-swap copy of the live file, so its bytes
   are still a prefix of it; that is the test now, and it runs on a stamp-less
   live file too. Every
   segment an answer actually depends on is still fully verified by the
   chain walk, and a named segment that is not a regular file or whose read
   fails is typed corruption, never a bare `OSError`.
10. **The lock has two tiers, by capability predicate** (round 4, made
    explicit in round 5): `platform_layer.kernel_file_locks_enforced` locks a
    scratch file in the lock directory once per process; only the kernel's
    own "this filesystem cannot" selects the name tier — exactly
    EOPNOTSUPP/ENOTSUP/ENOSYS *(correction, round 5.4: ENOLCK was in the set
    until then — "no locks available" is a missing lock daemon OR an exhausted
    kernel lock table, not a capability answer, and it selected the tier where
    the round-3 race returns; round 5.4 made it keep the enforced tier and fail
    EVERY live acquisition closed — product-wide, the lenses showed, since the
    primitive is shared — so the close-out made ENOLCK the name tier with its
    errno RECORDED and a per-caller refusal: only the monetary lock names it
    (`refuse_name_tier_errnos={ENOLCK}`), so a lockd-less NFS refuses every
    monetary write with `UsageAccountingError` instead of running the name
    protocol while every other lock keeps the protocol it always ran there;
    the per-directory verdict is decided ONCE under a module lock, so racing
    threads share one probe; an unprobeable directory answers enforced for
    that call, uncached — §9 "Round 5.4" and §10, R1)*, and since
    round 5.3 the two Win32 answers of a volume without byte-range locks —
    ERROR_INVALID_FUNCTION, which LockFileEx answers on `\\wsl$`
    (microsoft/WSL#5762), and ERROR_NOT_SUPPORTED, error 50 on a Samba share —
    map onto ENOSYS and EOPNOTSUPP (onto ENOLCK and EOPNOTSUPP before round 5.4):
    CPython's winerror→errno table lands both on EINVAL, which left the name
    tier structurally unreachable on Windows, so a lock-less volume there
    failed EVERY monetary append closed instead of degrading to it.
    *Enforced tier* — POSIX flock and Windows LockFileEx, both held on
    the lock fd — a refusal that is not contention fails the acquisition
    closed (no descriptor, our own file removed, a stale lock never evicted
    without the hold); a live-but-WEDGED holder can no longer be evicted by
    age — the deliberate trade of an availability incident (the wedged writer
    must die first) for the correctness incident (age-evicting a live
    monetary writer). Windows cannot unlink an open file: its eviction and
    release re-check the path after the close (release included — the POSIX
    "unlink under the still-held flock" is POSIX only, stated so since round
    5.4), and a freshly won lock — held open by its owner — is undeletable
    there. Disclosed since round 5.4, both tiers: a contention answer on the
    creator's OWN fresh file (a foreign flock holder that never unlinks it)
    leaves that live-pid-stamped file on the path — the creator re-contends
    against it until its timeout and owner-aware acquirers never age it out
    while the process lives; no in-protocol holder produces that shape.
    Also since round 5.4 the recycled-pid disclosure names its real
    mechanism: `pid_is_alive` read EPERM as DEAD, so another user's recycled
    pid was reclaimed through the age path (flock-guarded on the enforced
    tier) and only a same-uid recycle wedged — the opposite of what round 5.3
    wrote; EPERM now reads alive (the process exists) — in `pid_is_alive`, the
    one liveness primitive shared by every consumer (custody settlements, claim
    reclaims, staging reaps defer for such a pid too) — so ANY live impostor —
    same uid or another — wedges the lock from the 90 s staleness window
    (`usage_ledger._locked`, `stale_sec=90.0`) until it exits, the probe
    flock deliberately unconsulted while the pid reads alive (a mixed-tier
    name-tier holder has none); Windows also has no
    `dir_fd`/`O_DIRECTORY`, so its archive bound and anchor scan stay
    path-based (fail-closed on any OSError since round 5). *Name tier* —
    kernel-lockless filesystems only — keeps the O_EXCL name protocol with
    re-check-then-unlink eviction, a disclosed best effort with no kernel
    exclusion: the compaction pass refuses to run there
    (`usage_compaction.NAME_TIER_REFUSAL`: logged, throttled by the growth
    guard, and since round 5.2 written ONCE per process per data root as a
    typed `usage_ledger_compaction_refused` event — the cause the 20 MB
    tripwire now names; the round-5 claim that the tripwire "names the case"
    was false until then) while ordinary appends continue under the name
    protocol. Residual, disclosed: the tier is decided per process per
    directory, so a lockd that dies mid-run can leave one process on each
    tier until restart — the name-tier process never compacts, and it also
    evicts by NAME with no kernel hold, so in that mixed mode the round-3
    two-writer class returns for the enforced-tier process's heartbeat-less
    APPENDS, not only for compaction. Also since round 5.2: the acquisition
    is identity-checked on both tiers (an evicted, still lock-less creator
    re-contends instead of returning a descriptor; the owner pid is written
    before the lock), and on Windows only ERROR_LOCK_VIOLATION reads as busy
    — access-denied and sharing-violation fail the acquisition closed at
    once instead of re-contending until the 45 s timeout (unexecuted here,
    owed to the 3-OS matrix). The round-4 claim that the
    anchor's path-based reads "can only ever ADD a corruption verdict" was
    false: a directory swapped after the walk made the path-based scan FAIL
    to add the verdict it owed; corrected in round 5 (§9, finding 3).

## 6. Round 2 — adversarial wave disposition (fix-round base `e2801c52`)

Verdict of the wave: NEEDS FIXES, nine findings. **All nine accepted and
fixed** — this is the monetary authority, so nothing was argued away as
theoretical. Every fix carries a pin that was verified RED against the exact
mutation it claims to catch (the mutation harness reverts one behaviour and
reruns the suite), and the three pins the wave called weak were rebuilt.

| # | wave finding | disposition | fix | red-first pin |
|---|---|---|---|---|
| 1 | HIGH — a long pass can be robbed of the lock (`stale_sec=90`, no `owner_aware_stale`), and a prior owner can unlink the new owner's lockfile; the swap then replays a stale snapshot over a concurrently appended charge | **accepted, fixed both ways** | `_named_lock` acquires owner-aware (a live PID is never evicted by age) and yields a heartbeat (`platform_layer.refresh_exclusive_file_lock`, descriptor-targeted so a stolen lock is never refreshed for the thief) that the pass beats at each checkpoint; **and** the swap is refused unless the live bytes still equal the snapshot, re-read under the same held lock right before the rename | `test_monetary_lock_is_owner_aware_and_the_pass_heartbeats_it`; `test_append_between_snapshot_and_swap_aborts_instead_of_erasing_it` (injected append → pass returns `None`, the row survives, money = before + that row) |
| 2 | HIGH — archive durability: only the segment's own parent is fsync'd, and `_fsync_dir` swallows every error, including on POSIX | **accepted, fixed** | `_mkdir_fsync_chain` syncs every directory entry the chain creates (segment parent, `archive/`, data root); `_fsync_dir` raises on POSIX and is a no-op on Windows *by the platform predicate*, not by a bare `except` | `test_archive_directory_chain_is_durable_before_the_swap` (fsync'd inodes recorded and required BEFORE the swap); `test_posix_directory_fsync_failure_aborts_before_the_swap` |
| 3 | HIGH — the baseline validator accepts a rolled-back hash chain and forged seq/epoch provenance; the archive reader scrapes ids instead of validating | **accepted, fixed** | substrate validates the stamp: epoch, bounded `archive_rel`, 64-hex sha, closing counts (`folded + retained == source`, first seq 1, last seq == source rows), block↔header agreement (`group_count`, summed `folded_attempt_count`), and `pre_compaction_seq` uniqueness/monotonicity under a stamp only. The reader runs each segment through `_validate_records` and requires the chain's epochs to step down by one to a header-less epoch 1 | `test_repointing_the_header_at_an_older_segment_is_corrupt` (three epochs; both skip shapes); `test_baseline_header_counts_must_close`; `test_pre_compaction_seq_is_a_checked_provenance_claim`; `test_rehashed_segment_still_fails_the_ledger_structure`; `test_a_group_row_cannot_rejoin_the_block_after_it_closed` |
| 4 | MEDIUM — a warm segment cache hides a deleted or replaced segment | **accepted, fixed** | the cache hit additionally requires the file's `(ino, dev, size, mtime_ns)` fingerprint; a miss re-reads, re-hashes and re-validates | `test_warm_segment_cache_revalidates_the_file_it_cached` (delete, then rewrite, both after a warm read) |
| 5 | MEDIUM — "decimal exactness" is bounded by the ambient 28-digit context, and the self-check rounds the same way | **accepted, fixed** | sums run under `_exact_money` (`prec=60`, `Inexact` trapped), so a loss past even that aborts instead of approximating; the pin's oracle sums in its own wider context | `test_group_sums_survive_beyond_the_default_decimal_precision` (10²⁸ + 1 keeps its last digit; red-first showed the dollar vanishing) |
| 6 | MEDIUM — `archive_rel` is not bounded to the archive directory | **accepted, fixed with 3** | `usage_ledger.valid_archive_rel` (textual bound, substrate-owned) plus a resolved-path bound in the reader (defeats a planted symlink) | `test_archive_reference_is_bounded_to_the_archive_directory` (six shapes rejected by the validator; an existing, correctly hashed file outside the archive rejected by the reader) |
| 7 | MEDIUM — a corrupt live header reads as "never compacted" | **accepted, fixed** | `_live_baseline_header` raises `UsageLedgerCorrupt` on an unreadable or non-object first row; `None` now means only "a readable row that is not a stamp" | `test_unreadable_leading_row_is_typed_corruption_not_absence` (the CPL-5 join raises → UNKNOWN, never an orphan verdict) |
| 8 | LOW — the join primitive re-unions the whole archived id set per question | **accepted, fixed** | the union is cached by chain identity ((`archive_rel`, sha) per hop); the stat-checked walk still runs, so finding 4's guarantee is not traded for the cache | `test_archived_id_union_is_built_once_per_chain` (H questions → exactly one union build) |
| 9 | MEDIUM — three pins do not pin what they claim | **accepted, all three rebuilt** | crash pin injects at `os.replace` itself and asserts the segment is already on disk with the exact source bytes (a swap-before-archive reorder now fails it); the threshold pin proves the lock is HELD at the call rather than trusting the call site; the head-only pin contrasts one unmodified baseline block that validates at the head with the same rows rejected purely for position | `test_crash_at_the_ledger_rename_leaves_ledger_intact`; `test_reserve_path_compacts_only_past_config_threshold`; `test_baseline_header_is_rejected_by_POSITION_not_by_shape` |

Not changed by round 2, and deliberately so: the fold scope (§3 of the
design note), the per-group baseline shape the wave independently confirmed
preserves per-root enforcement and all five breakdown axes, the trigger
thresholds, and the ABI-3 allowlist row the wave found correctly scoped.

New residual disclosed by finding 1's fix: a pass that loses the snapshot
race leaves an orphan archive segment (already written, never referenced).
Orphan segments were disclosed as harmless before, and the archive is
append-only by design (§13); the alternative — swapping anyway — is the
defect being fixed.

Round-2 code commits (author `ouroboros-agent`, single-intent):
`9e99eb55` (findings 1, 2, 9-crash, 9-threshold), `0ed2dc2c` (findings 3, 4,
6, 7, 8, 9-position), `6b03212e` (finding 5 + the ARCHITECTURE ownership
line).

## 7. Round 3 — second adversarial wave disposition (fix-round base `830aa35a`)

Verdict of the second wave: NEEDS FIXES. It re-read the round-2 fixes and
judged five of the nine findings still OPEN (1, 2, 3, 4, 6), closing 5, 7, 8
and 9. **All five accepted and fixed**; nothing was argued away. Each fix carries a pin verified RED against the exact mutation it
claims to catch, on this base, before the fix landed.

| # | round-2 verdict | what was still open | fix | red-first pin |
|---|---|---|---|---|
| 1 | HIGH, OPEN | ownership was never actually proven: stale inspection judged the path and then unlinked the path; release unlinked whatever now occupied the name; the POSIX heartbeat renewed the descriptor and answered success after it had been unlinked; `_beat` ignored the answer; nothing beat during the long build/verify span; and the snapshot compare→replace stayed a TOCTOU window | `platform_layer` now compares descriptor identity with path identity everywhere: the eviction removes only the exact file it judged (re-checked immediately before the unlink), the release removes only the file it still holds, and `refresh_exclusive_file_lock` returns an OWNERSHIP verdict. `_beat` aborts the pass on a lost or unanswerable hold and runs inside both candidate row walks and between every verification stage. The window is closed structurally — every ledger writer takes this same owner-aware lock with no unlocked fallback — and the swap re-reads what landed | `test_stale_eviction_never_removes_a_lock_re_created_under_it`, `test_release_never_unlinks_a_lock_that_was_stolen`, `test_heartbeat_reports_lost_ownership_instead_of_renewing` (+ deleted-lock variant) in `tests/test_lockfile_helpers.py`; `test_a_lost_lock_aborts_the_pass_instead_of_swapping`, `test_the_long_build_and_verification_section_beats_the_lock`, `test_no_writer_can_append_between_the_snapshot_check_and_the_swap`, `test_every_ledger_writer_refuses_when_the_lock_cannot_be_taken`, `test_a_swap_that_did_not_land_is_a_typed_failure_not_a_receipt` |
| 2 | HIGH, OPEN | durability was established only for the levels a pass CREATED, so the retry after a pass that died on its own fsync skipped directories that already existed but were not yet durable | `_mkdir_fsync_chain(path, root)` fsyncs the whole chain up to the data root unconditionally, every pass | `test_the_directory_chain_is_re_synced_on_the_retry_after_a_failed_pass` (fails the first pass on the directory fsync, then requires all three inodes fsync'd before the retry's swap) |
| 3 | HIGH, OPEN | the chain had no trusted live anchor — `compaction_epoch` is as mutable as the rest of the row, so repointing the header at an older genuine segment AND lowering the epoch walked a valid short chain; `pre_compaction_seq` was only required to increase | the archive anchors the stamp: no segment may carry a generation newer than the live one, derived from each segment's embedded header (content, not name), with an uncommitted orphan of the live generation explicitly legal. `pre_compaction_seq` must fall inside the header's declared source range | `test_repointing_the_header_at_an_older_segment_is_corrupt` — the forgery now copies `compaction_epoch` too, which the wave named as the pin's escape hatch; `test_pre_compaction_seq_must_name_a_row_the_named_source_held`; `test_an_orphan_segment_of_the_live_generation_is_not_a_rollback` guards the fix against over-reach |
| 4 | MEDIUM, OPEN | a fingerprint is not identity: an in-place same-size rewrite inside timestamp granularity, or with the mtime restored, kept the cache hit | a hit also requires an mtime settled for > 2 s and an entry younger than 60 s; past either, the bytes are hashed again. Remaining window disclosed as residual §5.7 | `test_a_rewrite_inside_the_timestamp_window_is_re_hashed_not_recalled`; `test_a_same_size_rewrite_is_caught_once_the_cache_entry_expires` |
| 6 | MEDIUM, OPEN | a symlink AT `archive/usage_ledger` escaped the resolved-parent bound, because segment and directory resolve through the same link | neither `archive/` nor `archive/usage_ledger` may be a link, the resolved directory must be exactly the resolved root's archive path, and no segment may be a link; the reader calls it corruption, the writer aborts its pass | `test_a_symlinked_archive_path_is_refused_by_writer_and_reader` (both levels, reader and writer) |
| 5, 7, 8, 9 | CLOSED by the wave | — | unchanged | unchanged |

ARCHITECTURE and the design note carried absolutes the wave was right to
call out (`never robbed of it`, unqualified `bounded`). Both now state the
contract with its residuals: ownership is defended and its loss is
survivable; the archive bound is exact about symlinks; the cache window and
the orphan segments are named where the mechanism is described (§5.6–5.9).

Round-3 code commits (author `ouroboros-agent`, single-intent): lock
ownership in `platform_layer` + its pins; the pass consuming ownership
(heartbeat abort, span checkpoints, post-swap verify); the unconditional
directory chain; the archive epoch anchor + source-range provenance; the
segment-cache shelf life; the archive symlink bound.

## 8. Round 4 — third adversarial wave disposition (fix-round base `d7b487ab`)

Verdict of the third wave: NEEDS FIXES. It judged the round-3 ownership and
bound fixes still short of the contract in four ways — the exclusion itself
was still only a name protocol, ownership was not proven adjacent to the
decisions it licenses, the recheck→replace gap remained, and the symlink
bound was still check-then-use. **All four accepted and fixed**; nothing was
argued away.

| # | what round 3 left open | fix | red-first pin |
|---|---|---|---|
| 1 | exclusion rested on the O_EXCL name protocol: the stale eviction re-checked the inode and then unlinked the PATH (a pause between the re-check and the unlink lets a second reclaimer remove the first one's freshly won lock — two writers on one monetary authority), and the release had the same window between its look and its unlink | the lock fd HOLDS a kernel lock (`fcntl.flock`; `LockFileEx` on Windows) from acquisition; a stale lock is evicted only while flock-holding the very fd that was judged, with the path re-checked under that hold, and a release unlinks BEFORE its close, under the still-held flock. Windows (no unlink of an open file) and filesystems without kernel locks keep the re-check-then-unlink shape as a best effort chosen by the platform predicate — disclosed, never an exception swallowed *(correction, round 5: false at `13af62c5` — any `OSError` from the kernel lock selected the name shape, silently; fixed by round 5, finding 1)* | `test_two_racing_reclaimers_never_yield_two_holders` (both reclaimers herded into the check-to-unlink window; RED on the round-3 code with both returning descriptors); `test_heartbeat_after_an_atomic_swap_of_the_lock_reports_false` (the path never absent, so an existence check would renew; red against the utime-only mutation) — both in `tests/test_lockfile_helpers.py` |
| 2 | the pre-swap re-check and the rename were separated by the tmp write and fsync: a row appended in that gap was erased by the swap, receipt and all | `_write_bytes_atomic_fsync` takes a `precondition` evaluated after the temp bytes are durable, immediately before `os.replace` — the last instant the replace can still be refused; the compactor passes `_snapshot_intact`, so the pass aborts with the ledger (and the landed row) byte-identical | `test_an_append_between_the_recheck_and_the_replace_aborts_without_loss` (RED on `d7b487ab`: the row was erased and a receipt returned; now the pass returns `None`, the row survives, money = before + that row, no temp residue) |
| 3 | ownership was beaten through the span but not adjacent to the decisions: nothing proved the hold immediately before the snapshot re-checks, and nothing at all between the final re-check and the swap | `beat()` now runs immediately before EACH snapshot look: a hold lost at the archive write aborts before the post-archive re-check is even asked (its answer would be meaningless), and a hold lost after that re-check aborts before the replace — the proof before the swap was moved INSIDE the atomic replace by the verification pass (panel FIX_FIRST; see the verification block below) | `test_a_hold_lost_at_the_archive_is_seen_before_the_snapshot_is_trusted` (asserts exactly ONE `_snapshot_intact` call; the "remove the beat before the re-check" mutation makes it two — red against that exact mutation); `test_a_hold_lost_after_the_recheck_aborts_before_the_swap` (RED on `d7b487ab`: the swap ran) |
| 4 | the symlink bound was check-then-use: `_archive_dir_bounded` / `_segment_path` judged paths, then the write and the read re-resolved those paths — a link planted in between received the segment (writer) or served a foreign file (reader) | POSIX opens the chain root→`archive/`→`usage_ledger` `O_DIRECTORY\|O_NOFOLLOW` handle-to-handle and creates/opens the segment `O_NOFOLLOW` via `dir_fd`, fingerprinting and reading from the open fd; directory durability is fsync'd through the same held handles. The path-based checks remain as the early typed abort and as the Windows best effort (no `dir_fd`/`O_DIRECTORY` there), chosen by the platform predicate | `test_a_link_planted_after_the_writer_bound_check_cannot_receive_history` (RED on `d7b487ab`: the segment crossed the link and the swap completed); `test_a_link_planted_after_the_reader_bound_check_is_refused` (byte-identical copy behind the link — the hash cannot object, only refusing the traversal defends; RED on `d7b487ab`) |

Confirmed rather than changed: every writer of this ledger already takes the
same owner-aware lock with no unlocked fallback (round-3 pin stands; what
changed is that the lock they all take is now kernel-held), and the
post-replace re-read stays, now after the in-swap re-proof.

New/updated residuals (also §5): a live-but-WEDGED holder can no longer be
evicted by age on POSIX — the kernel lock outlives the staleness clock until
the process dies. That is the deliberate trade: age-evicting a live writer
was the two-writers defect; a wedged monetary writer is an availability
incident, not a correctness one. Windows and kernel-lockless filesystems
(bare NFS and friends) run the round-3 identity-re-check shape as a disclosed
best effort selected by the platform predicate *(correction, round 5: at
`13af62c5` the selection was by exception, not by predicate — §9, finding
1)*. `ouroboros/usage_compaction.py`
entered the 1001-1500 size band with a recorded rationale (the dir-fd
anchoring and the in-swap re-proof live beside the pass they defend);
`ouroboros/platform_layer.py` stays inside the band at 1498 lines, paid for
by prose compression in the same module.

### Round-4 verification (base `d7b487ab`; the round-4 work had shipped unexecuted)

Round 4 was authored in an execution-denied environment, so a dedicated
verification pass ran every claim for real. One finding of the round-4
review panel (codex, FIX_FIRST, accepted by the coordinator) was fixed in
the same pass:

- **The ownership proof stood before the swap, not inside it**: `beat()` ran
  immediately before `_swap_ledger_fsync`, but the atomic writer can spend
  arbitrarily long writing and fsyncing the candidate temp before its
  snapshot look and `os.replace` — a hold lost in that window let a new
  holder's charge (landing after the in-swap snapshot answer, before the
  rename) be erased by the swap. The proof of ownership now lives in the
  precondition of the atomic replace itself: once the temp bytes are
  durable, immediately before the rename, ownership FIRST and the snapshot
  compare only under a proven hold (`_swap_ledger_fsync` passes `beat` into
  `_write_bytes_atomic_fsync`'s precondition). Pin:
  `test_a_hold_lost_while_the_temp_is_written_refuses_the_replace` — RED on
  the round-4-as-authored shape (ownership died with the temp on disk; the
  snapshot-only precondition let the rename run and a receipt returned),
  green with the fix: the replace is refused and the new holder's charge
  survives byte-for-byte, money = before + that charge.

Every round-4 red-first claim was then observed, not argued — each pin was
run against the exact mutation or base it names (mutation applied, pin RED,
mutation reverted, pin green):

| pin | mutation | red observed |
|---|---|---|
| `test_two_racing_reclaimers_never_yield_two_holders` | `platform_layer.py` reverted to `d7b487ab` | both reclaimers returned descriptors: 2 holders |
| `test_heartbeat_after_an_atomic_swap_of_the_lock_reports_false` | identity comparison removed from `refresh_exclusive_file_lock` (utime-only) | heartbeat answered True for a replaced lock |
| `test_an_append_between_the_recheck_and_the_replace_aborts_without_loss` | swap precondition removed entirely | receipt returned; the injected row was erased |
| `test_a_hold_lost_at_the_archive_is_seen_before_the_snapshot_is_trusted` | post-archive `beat()` removed | 2 `_snapshot_intact` calls instead of 1 |
| `test_a_hold_lost_after_the_recheck_aborts_before_the_swap` | in-swap ownership proof removed (snapshot-only precondition, no outer beat) | the swap ran; a baseline landed |
| `test_a_hold_lost_while_the_temp_is_written_refuses_the_replace` | round-4-as-authored shape (outer `beat()` + snapshot-only precondition) | receipt returned while robbed |
| `test_a_link_planted_after_the_writer_bound_check_cannot_receive_history` | `usage_compaction.py` reverted to `d7b487ab` | the segment crossed the link; the swap completed |
| `test_a_link_planted_after_the_reader_bound_check_is_refused` | `usage_compaction.py` reverted to `d7b487ab` | the byte-identical copy was read through the link (no raise) |

Windows tier: the two new lockfile pins exercise POSIX mechanics (flock-held
eviction; replacing an open, kernel-locked file), so both carry
`skipif(IS_WINDOWS)` with the disclosed-best-effort reason; the compaction
pins are platform-neutral, and the two planted-link pins already skip on
Windows. `fcntl` is imported only inside `not IS_WINDOWS` branches of the
`platform_layer` primitives, so the module imports cleanly where `fcntl`
does not exist.

Round-4 verification gate evidence (this host, isolated env roots, venv
python 3.10.12 / pytest 9.1.1): recorded in
`docs/v7next/LEDGER_CORRECTIONS.md` §"From the C6 fix-round 4 verification"
— targeted usage/lockfile suites green; CI-shape non-serial battery EXIT=0;
`-m serial` EXIT=0; `-m size_ratchet` green; `ruff check . --select F`
clean; `scripts/check_domains.py` OK; `scripts/regenerate_inventories.py
--check` OK; `git diff --check` clean; `git rev-parse HEAD` verified after
every pytest run. With that run recorded, round 4 is verified, not merely
code-complete.

## 9. Round 5 — fourth wave disposition (fix-round base `13af62c5`)

Verdict of the fourth wave (gpt-5.6-sol, read-only): NEEDS FIXES. It closed
the round-3 split-brain class (stale eviction under flock with the inode
re-check, release of the own pathname only, identity heartbeat), all monetary
writers under one lock, and the archive writer/reader dir-fd anchoring; it
left four items open. **All four accepted and fixed**; nothing was argued
away. Each fix carries a pin verified RED against the exact pre-fix shape or
mutation it names, on this base, before the fix landed.

| # | what round 4 left open | fix | red-first pin |
|---|---|---|---|
| 1 | HIGH — on ANY `OSError` from the kernel lock the acquisition silently degraded to the pathname/inode name tier, where the round-3 race returns (and on Windows the errno-less `LockFileEx` failure fell into the same degrade) | the tier is an explicit capability predicate, `platform_layer.kernel_file_locks_enforced(lock_path)`: one scratch-file kernel lock per lock directory per process; only ENOLCK/EOPNOTSUPP/ENOSYS select the name tier. On the enforced tier contention (EAGAIN/EACCES/EWOULDBLOCK) stands down and re-contends; every other refusal fails CLOSED — no descriptor, our own file removed, a stale lock never evicted without the held flock. `_win32_lock` raises an `OSError` carrying the Windows error so `ERROR_LOCK_VIOLATION` classifies as contention. The name tier makes no kernel call at all, and `compact_usage_ledger_locked` refuses it with the typed `NAME_TIER_REFUSAL` (logged; appends continue under the name protocol, disclosed). `usage_ledger.LOCK_REL` is the lock-path SSOT *(correction, round 5.3: the busy set is EAGAIN/EWOULDBLOCK alone — EACCES fails closed since round 5.2, finding W — and the unsupported set is ENOLCK/EOPNOTSUPP/ENOTSUP/ENOSYS plus the two Win32 codes round 5.3 maps onto it; §9 "Round 5.3", findings 2 and L5; correction, round 5.4: ENOLCK left the set — it fails closed — so the set is EOPNOTSUPP/ENOTSUP/ENOSYS, with winerror 1 mapped onto ENOSYS; §9 "Round 5.4", R1)* | `test_a_kernel_refusal_that_is_not_contention_fails_closed`, `test_a_stale_lock_is_never_evicted_without_the_kernel_hold`, `test_the_name_tier_is_chosen_by_the_predicate_not_by_a_refusal`, `test_the_capability_probe_decides_once_and_leaves_no_residue`, `test_windows_lockfileex_contention_reads_as_busy` (skipif not Windows) in `tests/test_lockfile_helpers.py`; `test_the_pass_refuses_on_the_name_tier_while_appends_continue` |
| 2 | HIGH — the ownership→snapshot precondition ran once before `utils.replace_atomic`, which retries `os.replace` up to ten times with pauses on a Windows sharing violation: a charge appended (or a hold lost) between attempts was erased by the retry that landed | `replace_atomic(src, dst, *, precondition=None)` asks the precondition immediately before EVERY attempt, retries included, and returns False without replacing when refused; `_write_bytes_atomic_fsync` routes its ownership-first, snapshot-second proof through it. POSIX behaviour is unchanged (one syscall) | `test_a_refused_rename_re_proves_the_hold_and_the_snapshot_before_retrying[append]` / `[hold_lost]` (first attempt raises `PermissionError`, the intrusion lands, the second call never happens, the row survives / the ledger is byte-identical) |
| 3 | MEDIUM — `_no_newer_archived_epoch` walked the archive by pathname and turned `OSError` into "no evidence": a directory swapped after the safe chain walk could hide a newer generation and admit a forged rollback (the §5.10 claim was false) | `archived_attempt_ids` opens the `O_DIRECTORY\|O_NOFOLLOW` handle chain ONCE — after the live-header read, for the rest of the question *(round 5.2: a directory swapped before that open is the same power as deleting the newer segments, disclosed; a non-regular entry is skipped, not corruption)*; segment loads and the anchor scan open entries relative to that same held handle (one `_open_archive_entry` rule; path-based only where `dir_fd` is absent). An entry the scan cannot list, open or read is `UsageLedgerCorrupt`; a first row that reads but does not parse stays the disclosed torn-segment case | `test_the_epoch_anchor_scans_the_directory_the_chain_was_walked_in` (POSIX; a look-alike directory swapped in after the walk), `test_an_archive_entry_the_anchor_cannot_open_is_typed_corruption` (a dangling entry) |
| 4 | LOW — two surviving mutations (deleting the first commit-section beat; losing the hold between rename retries) and three doc absolutes («cannot finish while robbed», «a hold lost anyway abandons», «kernel-held») stated without their tier | both pinned (the second by finding 2's `[hold_lost]` variant); DESIGN §8/§10/§12, ARCHITECTURE and this packet now state the contract per tier — enforced tier vs name tier | `test_a_hold_lost_before_the_first_commit_look_writes_no_orphan` |

Red observed, not argued — each pin against the exact pre-fix shape or
mutation it names (pin red, fix applied or mutation reverted, pin green):

| pin | mutation / base | red observed |
|---|---|---|
| `test_a_kernel_refusal_that_is_not_contention_fails_closed` | `13af62c5` (silent degrade on any OSError) | a descriptor was returned for an ENOLCK-refused lock |
| `test_a_stale_lock_is_never_evicted_without_the_kernel_hold` | `13af62c5` (`evict_flockless` on a non-contention errno) | the stale file was evicted by name and a descriptor returned |
| `test_the_name_tier_is_chosen_by_the_predicate_not_by_a_refusal` | `13af62c5` (kernel lock attempted unconditionally) | a kernel call was made on the name tier (`[16] == []`) |
| `test_the_capability_probe_decides_once_and_leaves_no_residue` | `13af62c5` | no predicate exists (`AttributeError: _KERNEL_LOCK_TIER`) |
| `test_the_pass_refuses_on_the_name_tier_while_appends_continue` | `13af62c5` (no tier check in the pass) | a receipt was returned on the name tier |
| `test_a_refused_rename_re_proves_the_hold_and_the_snapshot_before_retrying[append]` | `13af62c5` (precondition once, `replace_atomic` retries blind) | the retried rename landed: receipt returned, the appended row erased |
| `…[hold_lost]` | same | the retried rename landed while robbed: receipt returned |
| `test_the_epoch_anchor_scans_the_directory_the_chain_was_walked_in` | `13af62c5` (path-based `iterdir`) | DID NOT RAISE: the look-alike directory hid epoch 3, the forged rollback passed |
| `test_an_archive_entry_the_anchor_cannot_open_is_typed_corruption` | `13af62c5` (OSError → continue) | DID NOT RAISE: the dangling entry was swallowed as no evidence |
| `test_a_hold_lost_before_the_first_commit_look_writes_no_orphan` | first commit-section `beat()` deleted | `[1] == []`: the pre-archive look was asked and an orphan segment written |

Windows tier, stated plainly: Windows ALREADY held `LockFileEx` on the lock
fd from acquisition (`file_lock_exclusive_nb` is platform-neutral); what was
missing was error classification, so the wave's suggestion of
`msvcrt.locking` was not adopted — it is a thinner CRT wrapper over the same
kernel lock with an EACCES/EDEADLOCK ambiguity, weaker than the `LockFileEx`
the module already owns. The Windows-only pin
(`test_windows_lockfileex_contention_reads_as_busy`, `skipif(not
IS_WINDOWS)`) and the `OSError(0, msg, None, winerror)` mapping it pins were
NOT executed on this host (Linux); they follow the documented CPython
constructor contract (errno derived from `winerror`, `ERROR_LOCK_VIOLATION`
→ `EACCES`) and stay disclosed as unexecuted until the 3-OS CI matrix runs
them. The four POSIX pins and the compaction pins ran here.

Size ratchet, stated plainly: `ouroboros/platform_layer.py` stays at 1498
lines inside the 1001-1500 band — paid for by prose compression in the same
module and by the pid lock and the port sweep reusing the module's own
primitives (`file_lock_exclusive_nb`/`file_unlock`, `force_kill_pid`), not
by any helper or neighbour module. `ouroboros/usage_compaction.py` grew
1094→1124 inside the band; its band rationale could NOT be extended — the
ratchet's own transition rule makes a surviving band rationale immutable
between adjacent manifests (`validate_manifest_transition`: "surviving band
rationale is immutable"), so the round-5 growth is recorded here and in the
ledger instead. `tests/test_usage_compaction.py` sits at 1492 inside the
band (the four copies of the raced charge folded into one `_raced_row`
helper paid for the new pins).

Round-5 code commits (author `ouroboros-agent`, single-intent): `f5eb969f`
(finding 1: lock tiers, fail-closed acquisition, name-tier refusal),
`8ed4f11b` (finding 2: the precondition before every rename attempt),
`a3d4d51d` (finding 3: the anchor through the held dir-fd, fail-closed),
`4b872c22` (finding 4: the first-commit-beat pin); the docs commit follows.
Gate evidence for this round is recorded in `docs/v7next/LEDGER_CORRECTIONS.md`
§"From the C6 fix-round 5 (base 13af62c5)".

### Round 5.2 — adversarial lenses over round 5 (fix-round base `2dd3e017`)

Verdict of the lenses (independent read of `2dd3e017`, PoCs executed on
scratch copies): five HIGH/MEDIUM findings open, six LOW. **All eleven
accepted**; nine are fixed in code with red-first pins, two are closed by
the disclosure the finding asked for (the doc absolutes; the mixed-tier
eviction residual). Nothing was argued away. Every code fix carries a pin
observed RED against the exact pre-fix shape or mutation it names.

| # | finding | fix | red-first pin |
|---|---|---|---|
| 1 | HIGH — a creator evicted while still lock-less flocks its own unlinked inode: between the O_EXCL create and the kernel lock the file is EMPTY (`owner_pid=0`, so owner-awareness cannot protect the window) and holds nothing an evictor must respect; stalled there past `stale_sec` (SIGSTOP, suspend, debugger, NFS clock skew) it is evicted, and its flock then SUCCEEDS on the unlinked inode — two descriptors believed to be one monetary lock (PoC `HOLDERS: 2`; the append transaction never heartbeats, so duplicate `seq` → a real charge quarantined). Same primitive with `stale_sec=10` and no owner-awareness at five non-monetary locks | the owner pid is written BEFORE the kernel lock, and a freshly won lock is returned only if the path still names the descriptor (one stat) — otherwise the creator closes it and re-contends. Both tiers, every caller of the primitive | `test_a_creator_evicted_while_lock_less_never_returns_a_descriptor` (`tests/test_lockfile_helpers.py`; the creator's first kernel lock ages its own file and runs an age-only reclaimer inline) |
| 2 | MEDIUM — the name-tier refusal was a throttled log line folded into the same `False` as "nothing foldable"; the "20 MB tripwire names the case" claim was false (the tripwire text named only a broken compaction or a large residue) | one typed `usage_ledger_compaction_refused` row per process per data root in `logs/events.jsonl` (the existing `append_jsonl`, contained like the compacted event; no return-type change); the tripwire text and the threshold comment name the third cause and the event; DESIGN §8, §5.10 and the module comment corrected | `test_the_pass_refuses_on_the_name_tier_while_appends_continue` (exactly one row after two refusals; the tripwire note names the tier and the event) |
| 3 | MEDIUM — «cannot finish while robbed» refuted in the last-proof→rename gap: `owned_and_intact` proved ownership, THEN read the whole file (≈1.8 ms on 8 MB), then `os.replace` — an fsync'd append (≈0.2 ms) by an out-of-protocol holder landed after the look answered True and before the rename (PROBE-1: receipt returned, row erased); the snapshot-first/beat-second mutation passed every pin | `owned_and_intact` beats, looks, beats AGAIN — the only interval between the last proof and the rename is the syscall (`replace_atomic` asks it before every attempt); DESIGN §8/§12, §5.8 and ARCHITECTURE state the bounded contract instead of the absolute | `test_a_hold_lost_after_the_last_snapshot_look_refuses_the_rename`; `test_a_hold_lost_after_the_recheck_aborts_before_the_swap` now also requires that the in-swap look is never asked once the hold is gone |
| 4 | MEDIUM — the anchor-swap pin pinned only the listing half: under "list via the held fd, OPEN BY PATH" it stayed green for the wrong reason (missing epoch-3 name → `FileNotFoundError` → "could not complete"), while a look-alike carrying the epoch-3 NAME with the forged live header as its leading row was ADMITTED by the orphan exemption | the look-alike now carries exactly that segment (forged header + the real epoch-3 body) and the pin requires `match="generation newer"` | `test_the_epoch_anchor_scans_the_directory_the_chain_was_walked_in` |
| 5 | MEDIUM — round 5's fail-closed rule made a stray subdirectory (an operator's `backup/`, no forgery) typed-corrupt for every history question forever (`os.read` → EISDIR); `13af62c5` answered. LOW siblings: a FIFO blocked the open indefinitely (pre-existing: neither fail-open nor fail-closed), and a directory standing where the header names a segment escaped as a bare `IsADirectoryError` the sweep's `except UsageLedgerCorrupt` would miss | `_open_archive_entry` opens `O_NONBLOCK` through the held dir-fd; `_no_newer_archived_epoch` fstat-classifies — a non-regular entry is no segment and is skipped, an entry it cannot list/open/read stays corruption; `_load_segment` raises typed on a non-regular named segment or any `OSError` of its fstat/read | `test_an_archive_entry_the_anchor_cannot_open_is_typed_corruption` (subdirectory + FIFO under a SIGALRM guard, then the dangling link with `could not complete`); `test_warm_segment_cache_revalidates_the_file_it_cached` directory shape (`not a regular file`) |
| W | LOW — the Windows busy set was a superset of contention: winerror 5/32/33 all land on EACCES, so a genuine access-denied re-contended until the 45 s timeout (latency only, no descriptor) | `_win32_lock_error` maps ERROR_LOCK_VIOLATION alone onto EAGAIN (winerror kept for diagnostics); every other Win32 error keeps its derived errno and fails closed; the busy set is {EAGAIN, EWOULDBLOCK} on both platforms | `test_lockfileex_refusals_classify_by_the_win32_error` (runs on POSIX too); the Windows-only contention pin keeps `winerror == 33` |
| D | LOW — doc absolutes and gaps: «cannot finish while robbed», «a hold lost anywhere abandons», «held for the whole question» (the handles open AFTER the live header read), the verbatim-restore rollback the orphan exemption admits, the mixed-tier residual omitting by-name eviction, §8 carrying the round-4 predicate claim without a marker, §5.9 contradicting itself | DESIGN §8/§10/§12, the ARCHITECTURE row, PERSISTENCE, §5.8/§5.9/§5.10/§8 of this packet rewritten as each finding asked; no code | — |

Red observed, not argued — each pin against the exact pre-fix shape or
mutation it names, on a scratch copy of this lane (pin red, fix applied or
mutation reverted, pin green):

| pin | mutation / base | red observed |
|---|---|---|
| `test_a_creator_evicted_while_lock_less_never_returns_a_descriptor` | `2dd3e017` | two descriptors returned (`[14, 15]`), `HOLDERS: 2` |
| `test_lockfileex_refusals_classify_by_the_win32_error` | `2dd3e017` | `EACCES in frozenset({11, 13})` |
| `test_a_hold_lost_after_the_last_snapshot_look_refuses_the_rename` | `2dd3e017` (beat → look → replace) | receipt returned while robbed, the charge erased |
| `test_a_hold_lost_after_the_recheck_aborts_before_the_swap` (look-count clause) | snapshot-first / beat-second (mutation M3) | `3 == 2`: the in-swap look was asked after the hold was gone |
| `test_the_epoch_anchor_scans_the_directory_the_chain_was_walked_in` | anchor opens entries by path (listing through the held fd kept) | `DID NOT RAISE UsageLedgerCorrupt` — the forged look-alike admitted; the previous pin shape passed under the same mutation |
| `test_an_archive_entry_the_anchor_cannot_open_is_typed_corruption` (subdirectory) | `79a1b9fb` | `anchor scan could not complete: [Errno 21] Is a directory` |
| same, FIFO half alone | `79a1b9fb` | `TimeoutError: FIFO open blocked` — the open hung until the 5 s alarm |
| `test_warm_segment_cache_revalidates_the_file_it_cached` (directory shape) | `79a1b9fb` | bare `IsADirectoryError: [Errno 21] Is a directory` from `os.read` |
| `test_the_pass_refuses_on_the_name_tier_while_appends_continue` (event + tripwire clauses) | `79a1b9fb` | no `events.jsonl` row at all (`FileNotFoundError`); the tripwire note named no tier |

Windows tier, stated plainly: `_win32_lock_error` and the classification pin
run their errno arithmetic on POSIX too (the new pin is not skipped), but
the LockFileEx call itself and the Windows-only contention pin remain
unexecuted on this host and owed to the 3-OS CI matrix; the path-based
Windows anchor scan keeps the fail-closed rule from round 5 (no
`S_ISREG`/`O_NONBLOCK` classification there — a directory in the archive is
corruption on Windows, disclosed), and the FIFO/dangling-link pin is POSIX
(`skipif(IS_WINDOWS)`).

Size ratchet, stated plainly: `79a1b9fb` (the round-5.2 agent's last
commit before the session limit) left `tests/test_usage_compaction.py` at
1512 lines with the manifest stale — `regenerate_size_ratchet.py --check`
exit 1 at that tree, the suite silently in the 1501-1600 zone; there is no
committed-history replay on this line (`review.py`: the local surface
warns), so the linear repair `6ad110e9` stands: three verbatim scaffolding
duplicates folded in place (the raced-charge-survived assertion, the
retry-durability pin re-running the first-pass proof, the single-caller lock
probe inlined) and PEP 8 spacing, 1512 → 1461, no new abstraction, every
folded pin still red under the swap-precondition-removed mutation; the
round-5.2 pins then bring it to 1492. `ouroboros/usage_compaction.py` grows
1124 → 1158 inside its band (immutable rationale, growth recorded here and
in the ledger); `ouroboros/platform_layer.py` 1499 and
`ouroboros/agent_startup_checks.py` 1490 stay inside theirs.

Round-5.2 code commits (author `ouroboros-agent`, single-intent): `847a1151`
(fold of the lock family's try/except-pass into `contextlib.suppress`, no
behaviour change), `7923e624` (finding 1), `f2b118a4` (finding W),
`ff6bb399` (one snapshot-look recorder for the hold/append pins),
`79a1b9fb` (finding 3), `6ad110e9` (the suite fold), `503a0dd6` (finding 5
and its LOW siblings), `95a53ad2` (finding 4), `208fe5ac` (finding 2); the
docs commit follows. Gate evidence: `docs/v7next/LEDGER_CORRECTIONS.md`
§"From the C6 fix-round 5.2 (base 2dd3e017)".

### Round 5.3 — adversarial lenses over round 5.2 (fix-round base `5e4829e3`)

Verdict of the lenses (independent read of `5e4829e3`, PoCs executed against
this lane's own code): six HIGH/MEDIUM findings open, seven LOW. **All
thirteen accepted**; twelve are fixed in code with red-first pins, one — the
recycled-pid wedge — is closed by the disclosure the finding itself offered
as its alternative (below). Nothing was argued away. Every code fix carries a
pin observed RED against the exact pre-fix shape or mutation it names.
The round ran in two halves: the first fix agent hit its session limit after
`cbfd23ce` with the docs staged and the ledger section unwritten; the resumed
half re-observed every red below in a scratch copy of this worktree before
changing anything, kept all ten commits, and closed two residues of the
round's own fixes (3b and 4b below).

| # | finding | fix | red-first pin |
|---|---|---|---|
| 1 | HIGH — `_lock_identity` answers `()` for a descriptor it cannot `fstat` (ESTALE/EIO — the network filesystems this tier exists for), and the acquisition compared the two identities RAW: with the path momentarily absent (a reclaimer's own unlink→re-create window) `() == ()` was vacuously true and a descriptor for an unlinked inode was returned as the monetary lock — `HOLDERS: 2`, and the ordinary append transaction never heartbeats. Second half: with the path present but the fd unstatable the bare `os.close` left our file stamped with our LIVE pid, which an owner-aware reclaimer may never evict — the lock wedged for the life of the process | the won lock is returned only when its own identity READS and matches; an unreadable one fails closed (`return None`, warning) and takes our stamp off the path when the bytes there are still exactly the ones we wrote. The stamp is captured once, at the write. The module's four other identity comparisons already guarded for the empty answer; `:268` was the outlier this round's own delta introduced | `test_a_lock_whose_identity_cannot_be_read_is_never_a_hold` (`tests/test_lockfile_helpers.py`; an fd-blind `_lock_identity` plus an evicting flock) |
| 2 | MEDIUM — the RATIFIED design note still called EACCES a contention code, the negation of the code, of round 5.2's own pin and of §5.10; implementing the note re-opens finding W (a genuine access-denied re-contending for the whole 45 s monetary timeout). The unsupported set was named three-of-four in three places | DESIGN §8 states both sets exactly (`EAGAIN`/`EWOULDBLOCK`; `ENOLCK`/`EOPNOTSUPP`/`ENOTSUP`/`ENOSYS`) with the Win32 answers that map onto them, and a pin compares the note's spelled sets with the code's, by number (EWOULDBLOCK/ENOTSUP are aliases on Linux, not everywhere) | `test_the_design_note_names_the_exact_kernel_refusal_sets` |
| 3 | MEDIUM — `heartbeat` defaulted to `None` and `_beat` returned at once on it, so a pass entered without one swapped the monetary authority with NO ownership check at all; the single production wire was unpinned (the reserve-path pin looked only at the lock, never at the kwargs), and deleting it (MUT-U) left the whole battery green | both entry points take `heartbeat` as a required keyword and `_beat` has no `None` case — a dropped wire is a TypeError at the call, not a silent no-op — and the reserve-path pin asserts the callable it is handed | `test_reserve_path_compacts_only_past_config_threshold` (`assert callable(kwargs["heartbeat"])`), red under MUT-U |
| 3b | LOW, own residue of 3 (resumed half) — the required keyword closes the dropped wire, but a caller passing `heartbeat=None` was unpinned: `_beat(None)` fails at the call and that failure IS the existing "answer we cannot get at all" abort, so the pass is refused — unproven by any pin | pinned beside the False and the raising heartbeats: `None` answers `None`, the ledger stays byte-identical, no orphan is written | `test_a_lost_lock_aborts_the_pass_instead_of_swapping` (`None` clause), red on the pre-finding-3 module |
| 4 | HIGH — the orphan exemption decided on ONE row: a segment whose leading row equalled the live header was an uncommitted orphan. The newest segment IS the previous generation's whole file, so a ledger restored from a backup taken just after that compaction satisfied it while being a strict SUBSET of the exempted segment — the attempts that pass folded exist nowhere else, and the join reported them absent (PoC: 4 of 8 ids hidden, no corruption raised, both segments on disk) | an orphan is the pre-swap COPY of the live file and the live file only grows behind it, so its bytes are still a PREFIX of it — that is the test, and it needs no live-id parse. A restored generation carries rows past the end of the file it was restored from | `test_a_restored_previous_generation_is_out_anchored_not_taken_for_an_orphan[stamped]` |
| 5 | MEDIUM — with the stamp itself gone (a pre-compaction backup restored) the anchor never ran: every gate sat behind `live_header is not None`, so `archived_attempt_ids` answered `frozenset()` having touched the archive zero times, and `_live_baseline_header`'s docstring claimed `None` "means exactly one thing" | the anchor runs either way, with the floor at epoch zero; a data root with no archive directory and no stamp still answers empty at once; the docstring names both states and points at the archive as the thing that tells them apart | `…[unstamped]` (same pin) |
| 4b | LOW, own residue of 4 (resumed half) — the prefix proof re-opened the entry by NAME after classifying it, so the bytes compared against the live file came from a second open: an entry swapped in between (an empty file; a writer-less FIFO under `O_NONBLOCK`) read as zero bytes and passed the anchor although the segment claiming the newer generation had just been read — the same power as deleting that segment before the scan (disclosed), but one open more than the proof needs | one open per entry: classify, parse the leading row and — when it claims a newer generation — `lseek` to the start and compare from the same descriptor (six lines fewer) | `test_a_restored_previous_generation_is_out_anchored_not_taken_for_an_orphan` (a second open of any name answers an empty file; the verdict must still be reached), red on the two-open shape |
| 6 | MEDIUM — `_archive_dir_fds` wrapped every open below the root and left the ROOT's own outside the `try`; `archived_attempt_ids` calls it with no handler, so an unreadable data root (permissions, EMFILE/ENFILE) left a bare `OSError` on the join surface — the class round 5.2 closed one function away | the root open is inside the `try` and typed `usage archive root is not readable` | `test_an_archive_entry_the_anchor_cannot_open_is_typed_corruption` (chmod `0o111` half) |
| L1 | LOW — the name-tier root was marked BEFORE the append, so one transient failure (ENOSPC, an unwritable `logs/`) turned the durable typed event back into a log line for the process's life, and the 20 MB tripwire then names an event that does not exist; the key was the unresolved path while the sibling growth guard resolves (two spellings of one root on this workspace) | the mark follows the row that landed; both maps key on the resolved root | `test_the_pass_refuses_on_the_name_tier_while_appends_continue` (failing append, then a landing one; then the same root through a symlink) |
| L2 | LOW — a lock whose owner died and whose pid was REUSED is never reclaimed (POSIX `kill(0)` answers EPERM for another user's process on this shared host), although the enforced tier's probe flock would settle it; `PERSISTENCE.md` still called these locks self-healing *(correction, round 5.4: the mechanism was misstated — `pid_is_alive` read EPERM as DEAD, so another user's recycle WAS reclaimed through the age path and only a same-uid recycle wedged; EPERM reads alive since round 5.4 and the disclosure names the real wedge — §9 "Round 5.4", R7)* | **disclosed, not changed** — the finding's own alternative. Taking the probe flock whenever the file is aged would also evict a LIVE holder of a mixed-tier install (the name-tier process holds no flock), trading a rare wedge for the two-writer class §5.10 already names; DESIGN §8, the ARCHITECTURE row and the PERSISTENCE row now state the wedge and the hand repair | — |
| L3 | LOW — the swap's own crash durability was unpinned on both sides: deleting the candidate temp's `fsync` (MUT-H) or the ledger directory's `fsync` after the rename (MUT-L) left the battery green, while the archive half carries three pins | one pin records the fsync'd inodes and the moment of the replace | `test_the_swap_fsyncs_the_candidate_before_the_rename_and_its_directory_after` |
| L4 | LOW — `_snapshot_intact` reduced to a size comparison (MUT-E) also left everything green: every intrusion the pins inject is an append | one intrusion rewrites a byte in place, changing no length | `test_a_same_size_rewrite_between_the_recheck_and_the_replace_also_refuses` |
| L5 | LOW — no LockFileEx refusal could select the name tier on Windows (CPython lands ERROR_INVALID_FUNCTION and ERROR_NOT_SUPPORTED on EINVAL), so a lock-less Windows volume failed every monetary append closed instead of degrading as disclosed; the classification pin's POSIX half asserted through errnos the function does not set | `_win32_lock_error` classifies by one table — 33 busy, 1/50 unsupported, anything else winerror-derived and fail-closed — and the classified codes carry their own errno (the 4-argument form derives errno FROM the winerror on Windows and ignores the one passed). Live evidence for exactly those two codes: LockFileEx on `\\wsl$` answers ERROR_INVALID_FUNCTION ("Incorrect function", microsoft/WSL#5762) and on a Samba share ERROR_NOT_SUPPORTED (error 50, "The network request is not supported", samba list thread "FileLockEx Problem") | `test_lockfileex_refusals_classify_by_the_win32_error` (1/50 must land in the unsupported set; 5/32/6 in neither) |
| L6 | LOW — "a stray directory or FIFO is no segment and is skipped" held only where the dir-fd exists: without one the entry was OPENED first, which a directory refuses on Windows (every history question typed-corrupt for one operator `backup/`) and a writer-less FIFO blocks on | without a handle the classification happens BEFORE the open, and the path-based open carries `O_NONBLOCK` where the platform has one | `test_an_archive_entry_the_anchor_cannot_open_is_typed_corruption[False]` (the path shape taken on POSIX, FIFO under the SIGALRM guard) |
| L7 | LOW — `_CHAIN_UNION_CACHE` was never bounded or expired although its key changes at every compaction, and DESIGN §10 still claimed the per-question cost was "H cheap walks and ONE union" after round 5 added the anchor | the map is bounded (only the newest chain can be asked again) and §10 states the anchor's per-question cost honestly | `test_archived_id_union_is_built_once_per_chain` (bound clause) |

Red observed, not argued — each pin against the exact pre-fix shape or
mutation it names, in this worktree (pin red, fix applied or mutation
reverted, pin green). Every row was re-observed by the resumed half in a
scratch copy of this worktree — the named source reverted to its base or
mutated as named, pin red; restored, the pins green together as one control:

| pin | mutation / base | red observed |
|---|---|---|
| `test_a_lock_whose_identity_cannot_be_read_is_never_a_hold` | `5e4829e3` | `assert 14 is None` — a descriptor for an unlinked inode returned as the lock |
| same, second half | the truthiness guard applied, the stamp cleanup not | `a live pid was stamped on a lock nobody may reclaim` |
| `test_the_design_note_names_the_exact_kernel_refusal_sets` | `5e4829e3`'s design note | `assert 0 == 2` — the note spelled no set at all (and called EACCES contention) |
| `test_reserve_path_compacts_only_past_config_threshold` | MUT-U (`, heartbeat=ledger_lock` deleted at `usage_accounting.py:706`) | `the pass was entered without the lock's heartbeat` (`{'heartbeat': None}`); without the pin that mutation left 136 passed, EXIT=0 |
| `test_a_restored_previous_generation_is_out_anchored_not_taken_for_an_orphan[stamped]` / `[unstamped]` | `5e4829e3` | `DID NOT RAISE UsageLedgerCorrupt` in both; the standalone PoC measured 4 of 8 ids hidden with both segments still on disk |
| `test_an_archive_entry_the_anchor_cannot_open_is_typed_corruption` (chmod half) | `5e4829e3` | bare `PermissionError` from `_archive_dir_fds`' root open |
| same, `[False]` (no dir-fd) | `5e4829e3` | `TimeoutError: FIFO open blocked` — the path-based open hung until the 5 s alarm |
| `test_the_pass_refuses_on_the_name_tier_while_appends_continue` (event clauses) | `5e4829e3` | `FileNotFoundError: …/logs/events.jsonl` — one failed append suppressed the event for good |
| `test_the_swap_fsyncs_the_candidate_before_the_rename_and_its_directory_after` | MUT-H (temp `fsync` deleted) | the candidate's inode absent from the fsyncs before the rename |
| same | MUT-L (`_fsync_dir(path.parent)` deleted) | the ledger's directory absent from the fsyncs after it |
| `test_a_same_size_rewrite_between_the_recheck_and_the_replace_also_refuses` | MUT-E (`_snapshot_intact` by size only) | a receipt returned: the swap landed over the rewritten row |
| `test_lockfileex_refusals_classify_by_the_win32_error` (unsupported clause) | `5e4829e3` | `assert (0 in frozenset({37, 38, 95}))` for winerror 1 |
| `test_archived_id_union_is_built_once_per_chain` (bound clause) | `5e4829e3` | `AttributeError: … has no attribute '_CHAIN_UNION_CACHE_MAX'` |
| `test_a_lost_lock_aborts_the_pass_instead_of_swapping` (`None` clause) | `usage_compaction.py` @ `c71a36ea^` (heartbeat defaulted, `None` skipped) | a receipt returned: `None` skipped every proof |
| `test_a_restored_previous_generation_…` (second-open clause) | `cbfd23ce` (the two-open anchor) | `DID NOT RAISE UsageLedgerCorrupt`, both parametrizations |

**Size ratchet — stated plainly, and an owner decision is now owed.**
`tests/test_usage_compaction.py` LEAVES the 1001-1500 band this round,
1492 → 1597, three lines under the 1600 HARD cap (the 1501-1600 zone is
ungated; above it the ratchet refuses new debt outright). Every commit of the
round carries a manifest that matches its own tree — the crossing commit
regenerates it — so the pairwise base-vs-tip lane is green at the tip and at
each parent, not only in the official CI shape.

The round asked for eight new or extended pins on the monetary authority; two
of them (MUT-H/MUT-L, MUT-E) close mutations that had survived the entire
battery. The ways to stay inside the band were: delete contract-bearing pins;
fold the five distinct `hold lost at X` pins into one table, merging the
per-moment reasoning each docstring carries; or add a neighbour suite — which
this lane's own band rationale rules out ("as one suite") and the owner's
standing rule forbids as payment for a cap. None was taken. What was paid
honestly: the retry-durability pin now calls the fsync-failure pin instead of
re-implementing it verbatim (−11); this round's own docstrings are compressed
with every claim kept and one no-cover `fstat` guard is gone (−6); the resumed half added six lines (the `None` clause, the second-open clause) and compressed the same docstrings once more (−4).
`ouroboros/platform_layer.py` stays at exactly 1500 — the two kill-tree
sweeps reuse the module's own `force_kill_pid` (−17) and `unlink_lockfile`
lost its `exists()`-then-unlink race (−2), which paid for the identity guard
and the Win32 table. `ouroboros/usage_compaction.py` 1158 → 1197, inside its
band (rationale immutable between adjacent manifests; growth recorded here
and in the ledger).

**The owner decision:** at 1597 the suite has three lines of headroom against
a cap that refuses new debt, so the NEXT pin on this surface cannot land
without one of — (a) splitting the CPL-5 join/history-reader pins into their
own suite (a real seam: a different module surface, a different consumer, its
own reason to change) against the recorded "as one suite" rationale, (b) an
authorized rebase of the ratchet baseline, or (c) accepting fewer pins on the
monetary authority. This round does not choose.

Round-5.3 code commits (author `ouroboros-agent`, single-intent): `7d134fd8`
(the kill-tree fold, no behaviour change), `7e6b935e` (finding 1), `f7b8a578`
(finding L5), `c71a36ea` (finding 3), `e08a0392` (findings 4 and 5),
`82250a45` (finding 6), `023b2e84` (finding L1), `232500f4` (finding L6),
`c5fa1ac7` (findings L3 and L4), `cbfd23ce` (finding L7, and the band crossing it pays for); resumed half: `48f7b115` (finding 3b), `72d17f51` (finding 4b); the docs commit —
DESIGN §8/§10/§12, the ARCHITECTURE row, PERSISTENCE, this section and the
ledger, with the design-note pin — follows. Gate evidence:
`docs/v7next/LEDGER_CORRECTIONS.md` §"From the C6 fix-round 5.3 (base
`5e4829e3`)".

### Round 5.4 — owner-bounded micro-round (base `096437c2`, owner batch №12, answer A)

Scope fixed by the owner: the residual list left by the Fable lenses over
round 5.3 and the independent gpt-5.6-sol read-only review — eight items, no
new exploration, no redesign. **All eight disposed**: seven changed in code
with red-first pins, one (R8 c–f) closed by the disclosures it asked for.
Every behaviour fix on the monetary path was observed RED on the pre-fix
shape before the fix landed (table below), then green with it.

| # | residual | disposition | fix | red-first pin |
|---|---|---|---|---|
| R1 | HIGH — `ENOLCK` sat in the unsupported set: "no locks available" is a missing lock daemon OR an exhausted kernel lock table, not the kernel saying this filesystem cannot, yet it selected the name tier — where the round-3 race returns; the per-directory tier cache was read and written with no synchronisation, so two first threads could run two probes and disagree | **fixed** | `_LOCK_UNSUPPORTED_ERRNOS` is exactly `EOPNOTSUPP`/`ENOTSUP`/`ENOSYS` (winerror 1 → `ENOSYS`, 50 → `EOPNOTSUPP`); ENOLCK keeps the enforced tier and a live acquisition the kernel refuses with it fails closed — no descriptor, our own file removed, no name protocol — so a lockd-less NFS refuses every monetary write typed (`UsageAccountingError`, the round-3 no-unlocked-fallback pin) and the pass is never entered; `_KERNEL_LOCK_TIER_LOCK` makes the probe single-flight: one probe, one verdict per directory per process. DESIGN §8, the ARCHITECTURE row and §5.10 spell the set (the design-note pin compares it by number) | `test_the_capability_probe_decides_once_and_leaves_no_residue` (ENOLCK clause; name tier now selected by EOPNOTSUPP), `test_enolck_keeps_the_enforced_tier_and_the_acquisition_fails_closed`, `test_two_threads_racing_the_first_probe_run_one_probe_and_read_one_tier` |
| R2 | MEDIUM — the root was marked "already told" whether or not `append_jsonl` landed the refusal row; the helper reports exhausted retries as `False`, not an exception, so one transient failure silenced the durable event for the process's life | **fixed** | the mark follows a `True` answer only; the failed append is logged by the helper and retried at the next refusal | `test_the_pass_refuses_on_the_name_tier_while_appends_continue` (a False-returning append between the raising and the landing one) |
| R3 | MEDIUM — the stamp-less fast path (`Path.is_dir()`) bypassed the typed root open: a regular file where the archive directory belongs answered a silent `frozenset()`, an uninspectable archive a bare `OSError` | **fixed** | before any compaction the question ends early only on the kernel's exact `ENOENT`; a non-directory or an uninspectable archive is `UsageLedgerCorrupt`; every case that reads anything then goes through `_archive_dir_fds`' typed root open as before (a plain `os.stat` classification first, because the Windows path shape has no dir-fd to route through) | `test_a_stamp_less_ledger_still_inspects_its_archive_fail_closed` |
| R4 | MEDIUM — bare `OSError` still escaped `archived_attempt_ids` through `pathlib`'s `is_symlink()` in `_archive_dir_bounded` (both levels) and `_segment_path` (the named segment): `pathlib` re-raises everything but ENOENT/ENOTDIR/EBADF/ELOOP, and an `archive/usage_ledger` readable but not searchable (a `chmod -R 600 data/` hardening) refused the segment's own `lstat` | **fixed, as a class** | both inspections wrap their `OSError` into `UsageLedgerCorrupt` ("cannot be inspected"), the same rule the opens follow since round 5.3 | `test_a_path_inspection_the_reader_cannot_make_is_typed_corruption` (the real chmod-600 shape, then `Path.is_symlink` raising `PermissionError`) |
| R5 | MEDIUM — DESIGN §8/§12.9, §5.8 here and the round-5.2 ledger line said a charge landed between the last proof and the rename is "erased, then surfaced by the post-swap re-read or quarantined at the next read"; neither could see it — the re-read compares the NEW inode against the candidate, the archive segment is the pre-row snapshot — so the loss was SILENT and a success receipt was returned | **fixed (POSIX) + docs corrected** | `_swap_ledger_fsync` holds the OLD inode open across the rename (the only witness left) and reads back whatever landed beyond the proven snapshot's length AFTER the fact: those bytes go to `state/usage_attempts.quarantine.jsonl` (`raw_base64`, the torn-tail shape, which flips `integrity_degraded`) and the pass raises `UsageLedgerCorrupt` instead of returning a receipt — never re-appended (`seq` belongs to the live file). Detected by size; Windows cannot hold the destination open through `os.replace` and stays a disclosed silent loss. The trigger's failure log no longer claims the ledger is uncompacted after such a raise | `test_a_swap_that_did_not_land_is_a_typed_failure_not_a_receipt[erased]` (the round-3 `[written_over]` variant unchanged) |
| R6 | MEDIUM — the production heartbeat wire was pinned as `callable(...)`: a constant-True stub (M9) survived the whole battery, every ownership proof a no-op | **fixed (pin)** | the reserve-path pin ages the lock file to the epoch, calls the heartbeat it is handed and requires `True` AND a renewed mtime — judged outside the contained call, so the red names the stub | `test_reserve_path_compacts_only_past_config_threshold` |
| R7 | MEDIUM — the recycled-pid disclosure named the wrong mechanism: `pid_is_alive` folded EPERM into "dead", so another user's recycled pid went down the age-eviction path (flock-guarded on the enforced tier) and only a same-uid recycle wedged — while DESIGN §8, PERSISTENCE and the round-5.3 L2 row said EPERM read alive | **fixed + disclosure corrected** | EPERM (the process EXISTS) reads alive; only ESRCH is dead; anything undeterminable reads present, as Windows already did — so `pid_provably_gone` is the exact negation of `pid_is_alive` and became one line (the reaper's docstring corrected with it). The real residual, stated in DESIGN §8, §5.10, ARCHITECTURE and PERSISTENCE: any live impostor — same uid or another — wedges the lock from the 90 s staleness window (`usage_ledger._locked`, `stale_sec=90.0`, a literal there, not a `config.py` constant) until it exits; the probe flock is deliberately not consulted while the pid reads alive (a mixed-tier name-tier holder has none) | `test_a_pid_that_refuses_our_signal_is_alive_and_its_lock_is_not_reclaimed` |
| R8a | LOW — `_build_candidate` defaulted `beat` to a no-op inside the monetary path | **fixed** | `beat` is required; omitting it is a `TypeError` at the call, contained by the pass, which then never compacts | control: `test_the_long_build_and_verification_section_beats_the_lock` red under M10 |
| R8b | LOW — the heartbeat's own failure modes were unpinned: the `not held or` guard (the heartbeat's analog of round 5.3 finding 1) and the `False` on a refused `utime` could each be removed with both suites green | **fixed (pin)** | the identity pin gains a refresh clause: a refused renewal answers False; an unreadable own identity with the path absent is not a match of two empty answers; a stranger's file at the path is not ours | `test_a_lock_whose_identity_cannot_be_read_is_never_a_hold` (refresh clauses) |
| R8c | LOW — a compaction committing between a question's live-header read and its anchor scan yields a transient false `generation newer` (UNKNOWN) | **disclosed** (DESIGN §10, §5.9, ARCHITECTURE) | the owner's scope for this item was disclosure; the bounded single retry the lens offered is not taken this round | — |
| R8d | LOW — stamp-less anchor consequences undisclosed: a ledger reset beside a surviving archive is a permanent `generation newer` verdict; a stray JSON file is corruption on a stamp-less ledger only | **disclosed** (PERSISTENCE ledger and archive rows' Reset columns, DESIGN §10, §5.9, ARCHITECTURE) | reset both together, or keep both | — |
| R8e | LOW — "an entry that is not a regular file … is skipped" was absolute; a UNIX socket cannot be opened at all (ENXIO) and reads as corruption | **qualified** (DESIGN §10/§12.5, §5.9, ARCHITECTURE) | an entry that OPENS but is not regular is skipped; one the kernel refuses to open at all is corruption. No socket pin: `AF_UNIX` paths are capped at 108 bytes and pytest's tmp paths exceed it, and a `chdir`-relative bind leaks process state into a parallel suite — a LOW not worth that hazard | — |
| R8f | LOW — two absolutes: "a release unlinks before its close, under the still-held flock" (POSIX only: Windows closes, then re-checks and unlinks) and no mention of the contention-branch orphan (an `EAGAIN` on the creator's OWN fresh file leaves a live-pid-stamped file the creator re-contends against and no owner-aware acquirer ages out) | **disclosed** (DESIGN §8, §5.10, ARCHITECTURE) | the orphan shape has no in-protocol producer (an evictor's flock unlinks what it judged): theoretical on the enforced tier, stated | — |

Red observed, not argued — each pin against the exact pre-fix shape or
mutation it names, in this worktree (pin red, fix applied or mutation
reverted, pin green):

| pin | mutation / pre-fix shape | red observed |
|---|---|---|
| `test_a_pid_that_refuses_our_signal_is_alive_and_its_lock_is_not_reclaimed` | `platform_layer.py` @ `bd9e99a4` (EPERM folded into dead) | `assert (False is True)` on `pid_is_alive(EPERM)`; the lock clause alone: the aged lock evicted, fd 3 returned, the file re-stamped with our pid |
| `test_the_capability_probe_decides_once_and_leaves_no_residue` (ENOLCK clause) | `platform_layer.py` @ `01c89685` (ENOLCK in the unsupported set) | `AssertionError: 37` — errno 37 selected the name tier (`False is True`) |
| `test_enolck_keeps_the_enforced_tier_and_the_acquisition_fails_closed` | same | `enforced = False`; a descriptor (fd 3) returned on the name tier, the lock file present, no kernel call made |
| `test_two_threads_racing_the_first_probe_run_one_probe_and_read_one_tier` | same (no cache lock) | `2 == 1`: both threads ran a probe |
| `test_the_pass_refuses_on_the_name_tier_while_appends_continue` (False-returning append) | `usage_compaction.py` @ `7ce7e83d` (mark regardless of the return value) | `FileNotFoundError: …/logs/events.jsonl` — the False answer marked the root, no row ever landed |
| `test_a_stamp_less_ledger_still_inspects_its_archive_fail_closed` | `usage_compaction.py` @ `b9c43911` (`is_dir()` fast path) | regular file: `DID NOT RAISE`, answered `frozenset()`; `archive/` chmod 000: bare `PermissionError` |
| `test_a_path_inspection_the_reader_cannot_make_is_typed_corruption` | same (bare `is_symlink()`) | `usage_ledger` chmod 600: bare `PermissionError: [Errno 13] … segment_ep0001_….jsonl` from the segment's own lstat; `Path.is_symlink` raising: bare `PermissionError` |
| `test_a_swap_that_did_not_land_is_a_typed_failure_not_a_receipt[erased]` | `usage_compaction.py` @ `d99ff6a9` (no old-inode witness) | `DID NOT RAISE`; standalone: receipt returned, the charge gone from the ledger, no quarantine file, `integrity_degraded` False |
| `test_reserve_path_compacts_only_past_config_threshold` (heartbeat clause) | M9: `heartbeat=lambda: True` at the production wire | `a stub, not the held lock's heartbeat` (`[False] == [True]`); the other 84 items of the two suites green under the same mutation |
| `test_a_lock_whose_identity_cannot_be_read_is_never_a_hold` (refresh clauses) | M15: the `not held or` guard removed from `refresh_exclusive_file_lock` | `assert True is False`: the blind descriptor renewed with the path absent |
| same | M16: a refused `utime` answers True | `assert True is False` |
| `test_the_long_build_and_verification_section_beats_the_lock` (control, R8a) | M10: `beat` dropped at the `_build_candidate` call | `TypeError: _build_candidate() missing 1 required positional argument: 'beat'`, contained by the pass → `assert None is not None` |

Disclosed, not fixed (with the reason each time):

1. **R1's consequence.** An install whose `state/` answers ENOLCK
   persistently (bare NFS without lockd) now refuses every monetary write
   closed — `UsageAccountingError` at the writer, one warning per attempt
   naming errno 37 — where round 5 ran the name protocol there. That is the
   owner's decision (fail closed, no name tier); the repair is a filesystem
   that locks. "Compaction refuses with a typed reason" is structural, not a
   new code path: with no lock the pass is never entered, and the round-3 pin
   `test_every_ledger_writer_refuses_when_the_lock_cannot_be_taken` is the
   typed refusal. sol's further suggestion — binding the established tier to
   the returned hold and passing that attestation to the compactor — is not
   taken: the module lock leaves one verdict per directory per process, which
   is the disagreement the attestation would have caught; a mixed-tier
   install ACROSS processes stays the round-5.2 disclosure.
2. **R2 and fsync.** No `events.jsonl` row is fsync'd, this one included;
   the mark is per-process memory that dies with the same crash that could
   lose an un-fsync'd row, so a new process re-tells. The residual — a
   delayed writeback error with the process alive (row lost, mark standing)
   — is the same for every event row and is not closed here.
3. **R5's bounds.** Detection is by size: a same-size in-place rewrite of
   the old inode inside the rename syscall is not a landed charge and is
   not seen. The erased bytes are preserved and flagged, never re-appended
   (a hand repair from the quarantine row). Windows: silent, disclosed.
4. **R7's trade.** The wedge now covers another-uid recycles too (they
   were reclaimed through the flock-guarded age path before); the
   alternative — the probe flock on any aged file — would evict a live
   name-tier holder of a mixed-tier install (round 5.3 L2 stands).
5. **R8c–f** are disclosures by the owner's scope: the transient UNKNOWN
   (no retry added), the reset-beside-archive verdict, the socket shape (no
   pin: the 108-byte `AF_UNIX` cap and a `chdir` hazard), the contention
   orphan and the POSIX-only release-under-flock.
6. **Windows** stays unexecuted on this host, as in every round.

Size ratchet, stated plainly: `ouroboros/platform_layer.py` 1500 → 1497
inside its band — `pid_provably_gone` folded to the one-line negation it now
is and the docstrings it gained reflowed, paying for `threading`, the tier
lock and the EPERM branch. `ouroboros/usage_compaction.py` 1197 → 1245
inside its band; the owner asked for the band rationale to be extended in the
same commit when the file grows, but the ratchet's own transition rule makes
a surviving rationale immutable between adjacent manifests
(`validate_manifest_transition`, "surviving band rationale is immutable"),
so — as in rounds 5, 5.2 and 5.3 — the growth is recorded here and in the
ledger instead. `tests/test_usage_compaction.py` 1597 → 1597: the round
added five new or extended pins (+67 lines) and paid with two no-behaviour
commits — a `compacted` fixture folding twenty-one verbatim seed-then-compact
preambles (−39) and argument-list/data-literal reflows within the file's line
width (−28); no claim, docstring, message or assertion was dropped, and no
neighbour suite was added. The three-line headroom under the 1600 hard cap is
what it was; the owner decision owed since round 5.3 still stands.

Round-5.4 commits (author and committer `Ouroboros`, single-intent):
`bd9e99a4` (the fixture fold, no behaviour change), `01c89685` (R7),
`7ce7e83d` (R1), `12558046` (R2), `b9c43911` (the reflow, no behaviour
change), `d99ff6a9` (R3 + R4), `ea4d4337` (R5), `9306f962` (R6),
`02338c9b` (R8a + R8b); the docs commit — DESIGN §8/§10/§12, the
ARCHITECTURE row, PERSISTENCE, this section and the ledger — follows. Gate
evidence: `docs/v7next/LEDGER_CORRECTIONS.md` §"From the C6 micro-round 5.4
(owner batch №12 A, base 096437c2)".

## 10. Round 5.4 close-out — three read-only lenses on `b4938c31`, operator disposition (owner batch №12 A)

Verdicts: 3 × NEEDS_FIXES, no HIGH; 3 MEDIUM + 7 LOW. Fixed here (base `b4938c31`), pinned red-first:

| finding | disposition | pin → pre-fix shape → observed red |
|---|---|---|
| MEDIUM R1 (two lenses) — ENOLCK fail-closed landed in the SHARED primitive: on a lockd-less NFS `state/` every `acquire_exclusive_file_lock` caller failed, no model call could dispatch; the owner decided "compaction refuses", not this | **fixed**: ENOLCK is the name tier with its errno recorded beside the verdict (`_KERNEL_LOCK_TIER[dir] = (enforced, errno)`); `acquire_exclusive_file_lock(refuse_name_tier_errnos=…)` lets a caller fail closed on a recorded errno; only `usage_ledger._named_lock` names ENOLCK. Ordinary locks keep the name protocol they always ran there; money refuses typed | `test_the_capability_probe_decides_once_and_leaves_no_residue` (ENOLCK clause) and `test_enolck_is_the_name_tier_for_ordinary_locks_and_a_typed_refusal_for_money` → `platform_layer.py`/`usage_ledger.py` @ `b4938c31` → `assert True == (True, 5)` (a bare bool cached, ENOLCK enforced) / `assert True is False` |
| MEDIUM R4 — `_segment_path` resolved with `Path.resolve(strict=False)` one line BEFORE the typed `is_symlink()`: a symlink loop escaped as `RuntimeError("Symlink loop …")`, a readlink race as bare `OSError` | **fixed**: `os.path.realpath` (non-strict, never raises on a loop) inside the same `try`, `except (OSError, RuntimeError)` → `UsageLedgerCorrupt` | `test_a_path_inspection_the_reader_cannot_make_is_typed_corruption` (self-loop clause) → `usage_compaction.py` @ `b4938c31` → `RuntimeError: Symlink loop from …` and `OSError: [Errno 40] Too many levels of symbolic links` |
| LOW R3 — the stamp-less ENOENT exemption used a FOLLOWING `stat`: a dangling link at either archive level answered a silent empty set where the stamped reader answers corruption | **fixed**: `lstat` both levels first; `S_ISLNK` → typed `usage archive path is a symlink`, other `OSError` → typed `cannot be inspected`; pin deferred (the compaction suite sits at its 1600-line cap, disclosed below); mutation-verified by hand on this host (dangling link at `archive/` → typed) | — |
| LOW R5 — the old-inode witness was opened by PATH before the proof and not tied to the inode the precondition proved; a vanished ledger at the witness open was a bare `OSError` | **fixed**: `owned_and_intact` also proves `fstat(old_fd)` and `stat(path)` name one inode; the witness open is wrapped into `_Abort` (an abort by policy) | — (behaviour-preserving strengthening; no pin, disclosed) |
| LOW R1 — DESIGN §8 "decides once … cached" was absolute; an unprobeable directory answers enforced UNCACHED | **docs**: DESIGN §8, packet §5.10/§9, ARCHITECTURE row | — |
| LOW R6 — the strengthened heartbeat pin proves renewal + True, not ownership: a lock-TOUCHING stub survives it | **disclosed, not fixed**: the pin proves the callable renews THIS lock file's age (the production wire's only observable) — a stub that touches the production lock path is a contrived mutation; the suite is at its line cap | — |
| LOW R7 — EPERM→alive is a flip of a primitive shared by 12 non-test consumers, disclosed only for the monetary lock | **docs**: DESIGN §8 residual, packet §9 R7, ARCHITECTURE/PERSISTENCE wording name the shared primitive and the consumers that now defer | — |
| LOW R8d — "PERMANENT … for the life of the install" over-stated: the verdict lasts until the fresh ledger's epoch passes the surviving segments, which are then silently ignored | **docs**: DESIGN §10, packet §5.9, ARCHITECTURE row | — |
| LOW R8e — the socket qualification introduced its own absolute: a UNIX socket is corruption on the dir-fd shape only; the path shape's stat-before-open skips it | **docs**: DESIGN §10/§12.5, packet §5.9, ARCHITECTURE row | — |

Sizes after the close-out: `platform_layer.py` 1500/1500 (band ceiling; net +3 on the policy, paid by rewrapping two prose blocks — no contract text dropped), `usage_compaction.py` 1262 (band), `tests/test_usage_compaction.py` **1600/1600** (the owner answered this in batch №13 item 11 = A: the archive-reader tests moved to their own module — the natural organ boundary, `archived_attempt_ids` vs the pass. After the split, the suite is `tests/test_usage_compaction.py` 900 + `tests/test_usage_compaction_archive.py` 660 + `tests/fixtures_usage_compaction.py` 123, same 64 node ids), `tests/test_lockfile_helpers.py` 568.

### §10 addendum — the Windows matrix (run 33654743857 on bf8b6549)

The lane never ran on Windows (not pushed until integrated). The first 3-OS matrix
after the merge was red on windows-latest only, in one class plus two test shapes:

- **Class (product):** the `LockFileEx` tier held a MANDATORY byte-range lock on the
  lock file, so a contender's `_lock_identity(probe)` read was refused and it could
  never judge the hold — `test_concurrent_writers_keep_monotonic_sequence` («usage
  accounting lock unavailable»), four `update_json_locked` timeouts, one lost
  concurrent chat append. **Disposition:** `kernel_file_locks_enforced` answers
  False on Windows — 7.0 ships Windows on the name tier it always ran (compaction
  refuses there, typed and disclosed); the tier code stays for the post-release
  re-enable with a stamp-safe byte range and a Windows-executed pin.
  **Correction (stage-2 delta review, lens e2e-and-ci; run 33663258606 on `35b82db0`):**
  the mandatory byte-range lock explained the bf8b6549 leg only; the same two tests
  (`test_concurrent_writers_keep_monotonic_sequence`,
  `test_terminal_projection_dedup_does_not_lose_concurrent_chat_append`) stayed red on
  every name-tier leg after it, because the name tier is NOT «the protocol it always
  ran»: since round 3 a contender opens the lock on every poll to read identity and
  owner stamp, and on Windows (CPython opens without FILE_SHARE_DELETE) that handle
  makes the owner's release unlink fail with a sharing violation — swallowed at debug,
  the lock is orphaned with the owner's LIVE pid, which no owner-aware acquirer evicts:
  the monetary lock refuses every later writer until restart, `append_jsonl` waits its
  2 s and lands unlocked (non-atomic append on Windows → lost rows). Reproduced on Linux
  by the verifier's delete-semantics simulator (1 refusal → orphan → 120 timeouts in
  20 s). **Fix:** `_unlink_lock_path` retries a transient Windows refusal for a bounded
  window at release and in `unlink_lockfile` (simulator: 288 refusals absorbed, 70 238
  acquisitions, no orphan); red-first pins
  `test_windows_release_retries_a_contenders_transient_sharing_refusal`,
  `test_windows_release_gives_up_a_refusal_that_never_clears`,
  `test_posix_release_does_not_retry_a_permission_refusal`. Verified by the matrix on
  the SHA carrying the fix (see LEDGER «From the Windows CI matrix on 35b82db0»).
  **Re-enabled in 7.0 by the Windows kernel-tier lane (commit `eb3ba7a1`), owner batch
  №13 item 1 = B: 7.0 does not ship until the kernel tier works.** The disposition above
  stands as history; what changed is the byte range. `_win32_lock` now holds ONE byte at
  `platform_layer._WIN32_LOCK_OFFSET` (`0x7FFFFFFF00000000`) instead of the whole file, so
  the stamp bytes [0, 512) a contender must read are outside every locked range;
  `kernel_file_locks_enforced` probes on Windows like POSIX and the compaction pass runs
  there (`tests/test_usage_compaction.py`'s `data_root` no longer skips). Windows eviction
  takes the same probe lock and unlinks after closing it — a WEAKER guarantee than POSIX's
  «at most one may evict», stated as such in DESIGN §8: the loser's unlink is refused by
  the winner's open handle, not by the kernel. Release order is unlock → close → unlink.
  Linux-side pins (`tests/test_lockfile_helpers.py`): the range constant and its two
  wrappers, an emulated LockFileEx refusing the same range while a contender still reads
  the stamp, eviction only under the probe hold, and the release order read off the fd's
  own liveness; plus the delete-semantics simulator (43 317 acquisitions in 20 s, 1 281
  sharing violations absorbed, no orphan). The Windows-EXECUTED proof is the next CI
  matrix — see LEDGER «From the Windows kernel-tier lane (owner 1 = B)».
- **Test shape (lane pins, POSIX protocol):** five lock-ownership pins unlink or
  rewrite a HELD lock file (impossible on Windows) and two swap pins assert
  directory fsync/inode identity — `skipif(IS_WINDOWS)` with the reason stated;
  `test_warm_segment_cache_revalidates_the_file_it_cached` accepts the path
  shape's typed refusal text.
- **Bystander:** `kill_process_on_port`'s POSIX branch, now routed through
  `force_kill_pid`, spelled `signal.SIGKILL`, which Windows lacks — the port-sweep
  tests drive that branch with `IS_WINDOWS` patched False; spelled portably.
