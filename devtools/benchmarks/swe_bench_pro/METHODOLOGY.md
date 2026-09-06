# SWE-bench Pro Methodology Notes

These notes summarize the portable lessons from prior Ouroboros CLI runs on
SWE-bench Pro (`scaleapi/SWE-bench_Pro-os`, dataset `ScaleAI/SWE-bench_Pro`,
images `jefzda/sweap-images:{dockerhub_tag}`, task repositories under `/app`).
They are not a replacement driver or scorer. They document how to prepare
prediction patches and how to inspect official Pro evaluator outputs without
repeating the same failure modes.

Included files:

- `capture_patch.sh`: standalone `model_patch` capture for a task repository.
- `pro_predictions.py`: capture predictions from already-solved prepared repos.
- `e1v2/`: the persistent-agent EVOLUTIONARY harness — solves instances in
  sequence with carried Ouroboros state/source volumes and native post-task
  evolution between tasks (see §3).
- `grade_pro.py`: wrapper that runs the official Pro eval and prints a
  diagnostic, non-leaderboard summary of official per-instance outputs.

## 0. Official SWE-bench Pro contract (authoritative — read first)

§1–§5 are Ouroboros *operational* notes (patch capture, our grading wrapper, the
stateful E1v2 variant, pitfalls); §6 is the anti-cheat rulebook. This section
states the **official** benchmark contract first, so the operational notes are
never mistaken for the protocol. Everything Ouroboros-specific — persistent
memory, native post-task evolution, retry-on-transient, the `scratch=[…]` verify
param, per-task volume reset, `strip_gold_history.sh` — is a **local scaffold
variant (§3 / §6 ALLOWED)**, NOT part of official SWE-bench Pro.

**Solver input (legitimate).** The task repository at `base_commit` (under `/app`)
plus the human-augmented description: `problem_statement`, `requirements`, and
`interface` (the signatures/contract), with repo name/language. The paper frames
each task as "complete with human-augmented problem statement, requirements and
interface."

**Evaluator-only metadata (NEVER legitimate solver input).** The dataset also ships
`patch` (gold fix), `test_patch`, `fail_to_pass`, `pass_to_pass`,
`before_repo_set_cmd`, `selected_test_files_to_run`, and `dockerhub_tag`. The gold
fix, the held-out tests and their names, future git history, and any hidden test
directory are **forbidden** solver inputs — surfacing any during solve invalidates
the score (§6 FORBIDDEN).

**Scoring (patch-based, resolved-iff).** The official Docker evaluator
(`swe_bench_pro_eval.py`): resets `/app` to `base_commit` → applies the candidate
`model_patch` → runs `before_repo_set_cmd` → runs the per-instance `run_script.sh`
over `selected_test_files_to_run` → parses with `parser.py`. An instance is
**resolved iff `(FAIL_TO_PASS ∪ PASS_TO_PASS) ⊆ passed_tests`** (every required
fail→pass test AND every guard pass→pass test passes). The evaluator restores the
gold/held-out tests itself ⇒ **local green ≠ resolved**; the solver must satisfy
`<requirements>` / `<interface>`, not the checked-out tests.

**Provenance & split (reporting context).** SWE-bench Pro is **1,865 problems across
41 repositories — 11 public / 12 held-out / 18 commercial** (paper abstract,
verified). The **731 public tasks** are the only publicly released problems (Hugging
Face `ScaleAI/SWE-bench_Pro`; independently confirmed — our
`helper_code/sweap_eval_full_v2.jsonl` = 731 rows); held-out and commercial problem
data are private (commercial *results* are published). Our 70-task subset
(`task_order_pro_70.csv`) is drawn from the 731 public rows.

**Integrity rules we enforce locally (not official defaults).** (a) Public Pro Docker
images have leaked future git history (official **issue #93**); we strip it before
solve (`strip_gold_history.sh`) — a *local* defense, not an official guarantee.
(b) We run with `web_search` (and the browser/vision/transcript tools) OFF via the adapter's
`--disable-tools` policy — that tool policy, not the network, is what actually keeps the
solver off the upstream fix. Treat it as *our* legitimacy choice: nothing official
requires it.
(c) **The solve container's network is OPEN, and egress isolation is NOT part of this
release.**
*What the official contract says about the solve network: nothing.* The official
SWE-bench Pro harness does not regulate the solve container's network at all — it
neither provides nor requires any solve-time network control. The `--block_network`
flag that does exist in the official evaluator applies to the **EVAL** container: in
`swe_bench_pro_eval.py` it is consumed only by `eval_with_docker`/`eval_with_modal`,
where it becomes `run_kwargs["network_mode"] = "none"` for the container that applies
the patch and runs the tests — verified in the official clone, not ours:
`SWE-bench_Pro-os/swe_bench_pro_eval.py:405`, flag defined `:464`.
*What we ship.* Solve containers get the **open, unisolated network** they had before
v6.75.0 — no network flags at all in the docker argv, no relay, no in-container reachability
probe, and no `--network-mode` flag on either driver. **Do not read anything in this
release as a network-level containment claim.** What actually keeps the solver off the
upstream fix is the tool policy in (b) (`--disable-tools`: `web_search`, browser and vision
tools off), plus the gold-history strip in (a) — both *our* legitimacy choices, neither
official.
*Why, and what was deferred.* A structural egress-isolation subsystem (an `--internal`
docker network plus one dual-homed fixed-upstream TLS pass-through relay per configured
provider host, with positive+negative in-container probes and a fail-closed refusal) was
built during v6.75.0 and then **extracted before release**; it is deferred to a later one,
so nothing in this tree implements it. The owner's decision (2026-07-25) that it would be
**off by default** is what made the extraction cheap: the comparison baseline we measure
against (codex) also runs with network access, so cheating is looked for **in the traces**
rather than prevented by the sandbox; and isolation would have cost us the musl/Alpine
instances outright, because their install-in-image transport needs general network during
setup.
*How many instances that would have been (measured 2026-07-25, not an estimate of
convenience).* The dataset itself cannot answer: the HF `ScaleAI/SWE-bench_Pro`
test split has 731 rows and 16 columns (`repo`, `instance_id`, `base_commit`,
`patch`, `test_patch`, `problem_statement`, `requirements`, `interface`,
`repo_language`, `fail_to_pass`, `pass_to_pass`, `issue_specificity`,
`issue_categories`, `before_repo_set_cmd`, `selected_test_files_to_run`,
`dockerhub_tag`) — there is **no libc, base-image or Dockerfile field**, and the tag
is an opaque `<repo>-<commit>` slug. libc is a property of the built image only, which
is why `run_pro.image_libc()` probes `/lib/libc.musl*` inside it.
Measured over every SWE-Pro instance ever solved on this host (287 distinct instances
across the local run archive; an install-in-image solve is identified by its
`/out/install.log`, which only that transport writes): **31 distinct instances are
Alpine/musl = 10.8% of the 287 attempted**, and **26 of those 31 (83.9%) produced a
non-empty patch** — i.e. isolating egress would not have been free, it would have removed
instances the agent was solving. musl is per-INSTANCE, not per-repo: the 31 span four repos
(`protonmail/webclients` 20, `gravitational/teleport` 9, `navidrome/navidrome` 1,
`future-architect/vuls` 1) with different Alpine bases per instance (v3.18, v3.21 seen
in the install logs), so a repo-level extrapolation (141 instances live in the two
biggest of those repos, 19.3% of 731) is an upper-bound guess, NOT the count.
Getting the exact 731-wide number is one pass of the existing probe over the dataset's
`dockerhub_tag`s — `docker pull jefzda/sweap-images:<tag>` then
`ls /lib/libc.musl*` (`image_libc`) for each row — which needs the network and roughly
a terabyte of image pulls; it is deliberately NOT run here. **These numbers are the
evidence the owner ruled on** (2026-07-25). Note precisely what they do and do not
license: they justify *not* imposing isolation (a change that can only ever restore
instances), and they are explicitly NOT a 731-wide count — no claim about the full set is
made on the strength of a 287-instance sample.

**Reporting rule.** Headline = **raw Pass@1 over submitted instances** (one patch per
task, first final answer; no manual patch repair, no audit-based re-weighting). The
formula is unchanged by v6.75.0. What is added is honesty about the denominator:
`grade_pro.py` classifies every instance as `pass` / `fail` / **`ungraded`** with a
typed reason (`no_official_output`, `output_unparseable`, `no_required_tests`,
`instance_not_in_dataset`), prints `ungraded=N/total` beside the unchanged headline,
and writes `grade_summary.json`. An instance the official evaluator never scored is not
a model failure. The `pass/(total−ungraded)` percentage is printed only when there ARE
ungraded instances and is explicitly labelled **diagnostic, not leaderboard-valid** —
it shrinks the denominator and must never be reported as the headline. Any
retry-on-transient resampling (§3.1) is disclosed with conditions + counts
(conditional best-of-N, not pure pass@1). `CONTAMINATION_AUDIT.md` is diagnostic-only
and never re-weights scores.

Sources: SWE-bench Pro paper <https://arxiv.org/abs/2509.16941>; official harness +
evaluator <https://github.com/scaleapi/SWE-bench_Pro-os>; dataset
<https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro>; image git-history leak
<https://github.com/scaleapi/SWE-bench_Pro-os/issues/93>.

## 1. Capturing `model_patch`

Patch capture determines what the official evaluator sees, and it is the most
common source of false failures.

- Capture like the reference SWE-agent/mini-swe-agent scaffold:
  `git add -A && git diff --cached <base_commit>`. A plain
  `git diff <base>` loses new untracked source files, and several real Pro
  fixes add files.

- Write the captured diff to an explicit path outside the Ouroboros repository,
  normally under `/Users/anton/Ouroboros/bench_runs/`. The helper rejects
  repo-internal output paths so benchmark artifacts cannot dirty `devtools/`.

- Remove environment artifacts that `git add -A` can capture. The `JUNK_RE`
  pattern in `capture_patch.sh` intentionally covers runtime dumps, caches,
  dependency folders, build outputs, coverage output, and similar generated
  files. Do not copy broad SWE-agent defaults such as `*.cfg`, `*.toml`, or
  `setup.py`: Pro fixes can legitimately touch configuration and manifests.
  Lockfiles are filtered structurally, not by extension: if a lockfile changes
  while its sibling manifest (`package.json`, `go.mod`, `Cargo.toml`,
  `pyproject.toml`, etc.) did not, and the patch also contains non-lockfile
  source changes, the lockfile is treated as installer/tooling churn and
  dropped. A pure lockfile-only patch is preserved.

- Remove binary blobs. `git diff --cached --numstat <base>` prints
  `-\t-\t<file>` for binary files. Build verification can leave compiled
  binaries in the repository; those can inflate a tiny source patch into a huge
  binary patch. Text additions such as `.go`, `.ts`, and `.py` files remain.

- The E1v2 container entrypoint calls the same `capture_patch.sh` helper mounted
  from `devtools/benchmarks/swe_bench_pro/`, so the persistent-agent path and the
  standalone prediction-capture path share one shell filter. The Python headless
  workspace-patch path applies the same lockfile-without-manifest rule separately
  because it serves real user/workspace artifacts, not just Pro benchmark diffs.

- In workspace mode, capture from the real task repository, usually `/app`, not
  from Ouroboros's internal repository. Verify that `git -C /app status` shows
  the intended modifications after the solve.

- Agent-created scratch files are the agent's responsibility, not a reason to
  over-filter patches. The helper filters environment artifacts and binary
  blobs, not arbitrary source-like files left by the agent.

## 2. Official Pro Eval And Diagnostic Summary

Run the official evaluator:

```bash
python swe_bench_pro_eval.py \
  --use_local_docker \
  --docker_platform linux/amd64 \
  --dockerhub_username jefzda \
  --scripts_dir run_scripts \
  --raw_sample_path <SWE-bench_Pro-os>/helper_code/sweap_eval_full_v2.jsonl \
  --patch_path patches.json
```

`grade_pro.py` wraps this command and then reads official per-instance
`{prefix}_output.json` files to print a diagnostic table. That table is not a
leaderboard result and is not a replacement scorer.

Important details:

- The Pro raw sample uses uppercase `FAIL_TO_PASS` and `PASS_TO_PASS` fields.
  Some Hugging Face-derived rows use lowercase names; handle both when
  inspecting diagnostics.

- If the official progress-bar accuracy aggregator fails or prints a misleading
  zero, inspect per-instance output files directly. The official evaluator
  output remains the source of truth; the local diagnostic only helps debug.

- Pro tamper protection restores test files from the fix commit after applying
  the agent patch. Agent edits to test files do not count as passing fixes.

## 3. Streaming Or Evolutionary Runs

The evolutionary harness is now `e1v2/`. Its hypothesis: solving instances in
sequence *with one self-improvement cycle between each* beats independent frozen
runs, because learned memory and reviewed self-modifications carry forward.

Scaffold defaults in the committed templates (v6.55.0, disclosed here):
`OUROBOROS_RUNTIME_MODE=pro` (the container is a disposable jail; unchanged),
`OUROBOROS_MAX_WORKERS=4` (was 5 — same-model subagent decomposition slots
within one instance, aligned with the TB/GAIA templates; NOT independent
attempts with selection), and `OUROBOROS_SAFETY_MODE=light` (the LLM safety
pass adds cost/latency inside an isolated jail while the deterministic guards
do the protecting; the LLM check stays for integration tools).
Every committed and derived fixed-model profile carries exactly one canonical
`api_model` row in `OUROBOROS_SUBAGENTS`, on the solve model. Thus decomposition
cannot inherit a default Light scout, second provider family, or agent-session
substrate even after the product's install defaults evolve.
`claude_code_edit` and `switch_model` stay in the default `--disable-tools` list:
benches measure the single-model Ouroboros harness. The retired edit name keeps
withholding its delegated coding successor through the task-contract compatibility
mapping; ordinary same-model `schedule_subagent` decomposition remains available.

E1v2 contract:

- The task repository is `/app` inside the official SWE-bench Pro image.
- `obo-data` and `obo-repo` volumes carry Ouroboros memory and self-modified
  source across tasks.
- (v6.56.0) The solve phase runs `/app` as the ACTIVE EXTERNAL WORKSPACE by
  default: the entrypoint passes `--workspace /app`, so contextual repo tools
  resolve against the task repo and the runtime records a workspace patch
  artifact alongside Method C capture (Method C from `/app` remains the
  OFFICIAL `model_patch` source; the workspace artifact is auditing/telemetry).
  Global improvement-backlog/promotion signals still flow from workspace tasks
  (v6.44.0), so native post-task evolution keeps working. Export
  `OBO_SOLVE_WORKSPACE_ROOT=""` for the legacy rootless `dig-direct` mode.
- (v6.56.0; policy spelling updated for ABI 7.0 — the `until_deadline` alias
  was removed, and the explicit pass cap was always the binding count axis)
  Budget/pacing disclosure: the solve task carries
  `budget_profile = {improvement_policy: fixed, max_improvement_passes: 6, cost_hard_stop_pct: 0}`
  — NO in-task cost stop (cost milestones are informational against the start
  snapshot); the bounds are the task deadline (`OBO_SOLVE_TIMEOUT`), the round
  ceiling (`OUROBOROS_MAX_ROUNDS=200`, below the leaderboard-relevant 250-step
  norms), and the per-task budget reset between instances.
- (v6.56.0) Per-task memory defaults to an EMPTY child drive
  (`OBO_MEMORY_MODE=empty` in the entrypoint): the measured artifact is the
  harness on each task, not memory accreted across tasks. Stateful/evolution
  runs export `OBO_MEMORY_MODE=shared` explicitly and disclose it.
- Patch capture uses Method C: `git add -A` then `git diff --cached <base_commit>`
  with validated junk/binary filters.
- The benchmark-only evolution steer asks for exactly one reviewed commit and a
  restart, AND in this environment forbids release/version bookkeeping (no
  VERSION/CHANGELOG/README/ARCHITECTURE/pyproject edits, no P9 version-bump
  rule). The standing steer already forbids touching those files, so advisory
  review routinely emits a `version_bump`/`forgotten_touchpoints` finding; that
  is expected and is left advisory. The review enforcement mode is
  owner-controlled — the steer is NOT "resolved" by hardcoding those findings to
  block (BIBLE P3). This prevents the self-hardening deadlock where every
  evolution commit becomes uncommittable under advisory mode.
- E1v2 templates deliberately stay `OUROBOROS_REVIEW_ENFORCEMENT=advisory` while
  every other v6.55.0 bench template runs `blocking`: E1v2 is the only bench
  with an in-bench `commit_reviewed` lane (the evolution cycle), and under
  `blocking` the steer-mandated absence of a VERSION bump becomes a
  conditionally-critical triad finding that blocks the commit — the evolution
  contract ("one reviewed commit, then restart") would be structurally
  uncommittable (v6.55.0 scope-review finding).
- The bench-local "Option A" heal for dangling evolution transactions is kept as
  a belt-and-braces in `entrypoint_pro.sh`: at task start it marks a committed
  transaction restart-verified at the container boundary (with a
  `git merge-base --is-ancestor` guard that ABANDONS a rolled-back commit
  instead). A current core's own boot reconciliation + supervisor auto-restart
  makes it a no-op; on agents seeded from an older core it prevents a poison-pill
  that wedges enqueue for every later task (E1v2 → E1).
- `owner_chat_id` is seeded into `state.json` BEFORE the per-task budget reset.
  The reset's load-modify-write creates `state.json` with only zeroed budget
  keys on a fresh volume; seeding after it would leave `owner_chat_id` unset and
  silently disable native post-task evolution (E1v2 would equal E0).

Stateful runs introduce failure classes that frozen baseline runs do not have.

- The physical-attempt ledger remains append-only across the campaign. Each
  instance has a distinct root task id and a $50 root limit, while the driver
  advances one cumulative campaign ceiling (up to $500 for the five-task wave).
  `reset_per_task_budget` clears only legacy compatibility counters; it never
  deletes or rewrites physical attempts.

- Count API errors by structured event type, not by substring occurrences inside
  nested provider messages. Separate transient transport failures from
  context-overflow recovery.

- Workspace mode often needs `memory_mode=forked`; shared memory can be
  forbidden with an external workspace. Verify that canonical parent reflections
  still grow across tasks.

- If task N has an infrastructure failure, restore state to the snapshot after
  the last clean task and rerun the suffix. Keep per-task snapshots of runtime
  data and source state.

### 3.1 Retry-on-transient policy (report this honestly)

`auto_run.py` resamples a task when its result looks like an infrastructure
transient rather than a genuine model failure. The accept gate is
`ok = (patch_bytes is not None) and (patch_bytes > 0 or genuine_0b or permanent_skip)`:

- a **non-empty** patch is always accepted;
- an **empty (0-byte)** patch is accepted as a *genuine* model failure only when
  `genuine_0b` holds — the agent actually ran (`n_events >= 2`), the provider
  channel was healthy (`api_errors < 3`), and the run was not cut short by infra
  (no OOM kill, no host `TIMEOUT`);
- a **`permanent_skip`** (a known non-recoverable environment fault:
  `pyexpat_abi_mismatch` / `server_import_failed` / `pip_bootstrap_failed` /
  `libc_skip`) is recorded as a non-run and is NOT retried;
- **everything else** is a transient — an empty patch with API errors or too few
  events, an infra-cut run, or a task that did not execute at all (`patch_bytes`
  is None, e.g. an image-unavailable or musl env-volume skip): the
  `obo-data`/`obo-repo` volumes are rolled back to last-good and the SAME task is
  re-run after `--retry-wait` (default 300s), up to `--max-retries` (default 5).

Two consecutive tasks exhausting their retries stop the shard (provider/infra is
likely down). Because each retry restores memory volumes, it is a fresh sample.

This must be disclosed in any results write-up: it is best-of-N **conditioned on
infrastructure-transient failures** (an empty patch with ≥3 API errors or too
few events, an infra-cut run, or a non-run), not pass@1. An empty patch that the
gate classifies as `genuine_0b` — the agent ran to ≥2 events with <3 API errors
and no infra cut — is kept as a real model failure and is NOT resampled, so the
resampling is confined to genuinely infra-degraded attempts; still, under routine
429 rate-limit spikes an attempt pushed over the API-error/infra thresholds can
be resampled, which can inflate patch/resolve rates on the affected subset. State
the retry policy, `--max-retries`, and how many instances were resampled. The
secret-opt-in gate: a task refused for missing opt-in is classified by the same
`ok` gate; treat a refused/empty run as not-resolved when reporting, not as a
successful sample.

`api_errors` counts real physical failures: a transport death inside one
request (the socket died after dispatch) produces one `llm_api_error` per
physical attempt. One call has one outer attempt budget
(`OUROBOROS_TRANSIENT_RETRY_MAX`), within which up to three rows can be typed
transport-death failures (the first death plus at most two repeats by the
primary dispatch) — so a flaky uplink can cross the `< 3` threshold on its
own and legitimately trigger a resample.

### 3.2 Seed provenance

The agent under test is seeded from the mounted source (`/opt/ouroboros-ro` →
`cp -a` into `/obo-repo`). Mount a clean checkout at a known tag: a dirty working
tree would leak uncommitted local edits into the measured agent and make the run
non-reproducible. Record the exact seed commit/tag with the results.

**v6.75.0 makes all of that enforced rather than advisory. Three DIFFERENT identities are
attested, never conflated:**

1. **the seed stamp** — `<VERSION> <HEAD>` of the mounted source, written into
   `/obo-repo/.git/ouroboros_seed` *while the volume is being seeded* (inside `.git`, so it
   is not untracked working-tree dirt). The seeding condition `[ -e /obo-repo/.git ] ||` is
   unchanged: an existing `.git` is an EVOLVED volume and is never re-seeded. The mounted
   HEAD is read ONCE and reused by both the stamp write and the verification, and an
   unreadable HEAD is refused as `seed_head_unreadable` rather than persisted as a
   sentinel: a stamped `unknown` would later read as `seed_mismatch`/`lineage_broken` and
   poison every subsequent task on that volume.
2. **the live HEAD of the evolving volume** — compared to the stamp by
   `git merge-base --is-ancestor`, not by equality: an evolution run legitimately moves HEAD
   forward, and an equality check would kill every task after the first.
3. **the version the running server reports over HTTP** (`GET /api/health`, frozen contract,
   read-only here) — compared to the live `/obo-repo/VERSION`.

Any mismatch refuses the task before the paid solve with a typed reason
(`stamp_absent` | `seed_mismatch` | `lineage_broken` | `runtime_skew` |
`runtime_unreachable` | `seed_head_unreadable`), `exit 88`, and a
recorded `/out/seed_attestation.json` that `run_pro.py` folds into the timeline row. All
five are properties of the VOLUME plus the mounted seed, identical for every task in the
shard, so BOTH drivers treat them as CAMPAIGN-fatal instead of per-task skips — recording
them per task would restore the same broken volume N times, burn the whole wall clock and
still exit 0 with a zero headline. The reason set is ONE shared authority,
`common/manifests.py::CAMPAIGN_FATAL_PROVENANCE_REASONS`: `auto_run.py` stops the shard (one
`FATAL: provenance refusal` line, exit 2) and `run_pro.py` stops its own schedule the same
way (exit 2, typed `volume_provenance` refusal in the run manifest), so a direct `run_pro`
invocation — or `orchestrate_probe.py`, which shells out to it — can no longer refuse every
task and report success.
Per-task refusals (`libc_skip`) stay per-task skips. The named override `OBO_ALLOW_EVOLVED_VOLUME=1` authorises EXACTLY ONE of
those reasons — `runtime_skew`, the deliberately evolved/version-skewed runtime it exists for
— and records `overridden: true` next to `override_set: true`: an audited exception, never a
silent one. It does NOT waive the others. A volume with no stamp, a foreign seed, a broken
lineage, an unreadable seed HEAD, or a `/api/health` that is unreachable or does not answer
with the contract (`runtime_unreachable`) means the provenance was never established at all,
so those stay fail-closed with the override exported; waiving them would let a paid solve
proceed while attesting nothing. The shell allowlist is a single token
(`OBO_OVERRIDABLE_SEED_REASON`) asserted equal to `manifests.OVERRIDABLE_ATTESTATION_REASONS`,
so the in-container and host-side gates cannot drift. The check is a ONE-SHOT step, deliberately not
inside `ready_probe()`: the readiness loop reads any non-zero probe rc as "not ready yet",
so a refusal there would be swallowed and burn the whole 900s window.

The host side is gated too: `run_pro.py` / `auto_run.py` build the run manifest immediately
after argument parsing, which is where the shared clean-seed gate runs
(`benchmark_run_manifest(require_clean=True)`), so a dirty or unidentifiable seed stops the
run before the first task instead of being discovered in a `-dirty` manifest afterwards;
`--allow-dirty-seed` records the exception. Unknown cleanliness is not cleanliness: when the
`git status` probe itself fails (corrupt `.git/index`, or its 10s timeout on a huge untracked
tree / CephFS) the record says `status_available: false` and the gate refuses with
`seed_status_unavailable` — the earlier coercion to `dirty: false` would have let a genuinely
dirty seed through with `seed_gate.ok: true`. Both drivers also assert the mounted seed has a
real `.git` **directory** — a `git worktree add --detach` seed has a pointer *file*, which
`cp -a` copies verbatim and which leaves the container with no git identity at all (no
stamp, no lineage, no self-edit accounting).

## 4. Container And Environment Pitfalls

- glibc runtimes mounted into Alpine/musl task images may not run. Use a
  compatible runtime build or glibc-based images when available.

- Readiness checks need wall-clock limits. Dependency installation under
  emulation can block `/api/state` for several minutes, and not every image has
  `curl`; a Python readiness probe from the agent environment is often more
  portable.

- Do not wait for heartbeat files such as `state/queue_snapshot.json` to become
  quiet. Watch durable outputs such as task reflections or task result files.

- On macOS bind mounts, host-side files can lag behind container writes. For
  live monitoring, read files inside the container with `docker exec`.

## 5. Debugging Checklist

1. Is the patch size reasonable? Huge patches often mean binary blobs; zero-byte
   patches often mean the wrong workspace was captured.
2. Inspect the `*.status.txt` emitted by `capture_patch.sh`.
3. Check raw sample field casing for `FAIL_TO_PASS` and `PASS_TO_PASS`.
4. Compare per-instance `{prefix}_output.json` files to see exactly which tests
   are missing.
5. Confirm that the agent did not rely on test-file edits.
6. Classify API errors by event type and failure class.
7. In stateful runs, check startup budget state before blaming solve quality.

## 6. Benchmark Legitimacy & Anti-Cheat (read before changing the harness or prompt)

This section is the durable rulebook so we never again propose an *illegitimate* harness
move (the recurring temptation is "let the agent peek at git history for the fix" or "show
it the tests"). The line is simple: **you may engineer the SCAFFOLD freely; you may not give
the agent the answer or the held-out oracle.**

### How grading actually works (why local green is not the oracle)

The official Pro evaluator does NOT trust the in-repo tests the agent sees during solve. It
resets the repo to `base_commit`, applies the model patch, then runs the task's **per-instance
setup/run scripts** — `before_repo_set_cmd` plus `run_script.sh`/`parser.py` (the evaluator's
CSV columns) — which **restore the gold/held-out test files** before scoring `FAIL_TO_PASS` /
`PASS_TO_PASS`. (`test_patch` in the dataset is the diff representation of those restored
tests; it is NOT a plain global `git apply test_patch` step.) See §2 "Pro tamper protection
restores test files from the fix commit." Consequence: **a local green can still be graded RED** — the agent must
derive correct behavior from `<requirements>`/`<interface>` (the contract), not from whatever
tests are currently checked out. This is why `e1v2/prompt_baseline.txt` discloses it
(legitimate scaffold disclosure — it reveals the grading *mechanism*, never the answer).

### ALLOWED — legitimate scaffold (Bring-Your-Own-Harness)

The SWE-bench contract is the SUBMITTED PATCH (`instance_id` / `model_name_or_path` /
`model_patch`); the grader applies it and runs the tests in Docker. The *inference harness* is
NOT standardized — system/developer prompt, tool/action schema, parser hints, memory/context
management, retry, candidate selection, and stopping rule are legitimate scaffold choices
(SWE-agent frames these as Agent-Computer Interface design and reports an ~10.7-point SWE-bench
Lite gain from ACI/scaffold alone, identical weights). So all of the following are legitimate
and are how harnesses legitimately differ on the leaderboard:

- Prompt discipline (e.g. "local green is not the acceptance oracle", "conform to `<interface>`
  exactly", "ground on the real test runner, not a `grep` proxy").
- Tool robustness (e.g. the `verify_and_record` argv/PATH fix), verification grounding, a
  finalize-gate that makes the agent reconcile its OWN red check before claiming done.
- Memory, retry-on-transient (disclosed — see §3.1), per-task reset, parallel namespacing.
- Reading ONLY public task info: the instruction text, embedded examples, installed oracles,
  and the agent's own independently-authored checks.

### FORBIDDEN — documented cheating (do NOT propose these)

- **Git-history mining for the fix.** Finding the fix commit via `git log`/`git log --all`
  (also future tags, dangling commits, reflogs, remote refs, latest upstream) and copying the
  historical/gold patch — or lifting the changed signatures from it — is the classic SWE-bench
  exploit (canonical report: SWE-bench issue #465, Sep 2025; maintainers scanned trajectories,
  adapted the images, and report it fixed for SWE-bench Verified). Our harness already strips
  gold history before the agent starts (`strip_gold_history.sh`, neutralizing SWE-bench Pro
  issue #93) and uses shallow/base checkouts. **Never re-introduce git-history mining as a
  "capability."**
- **Showing or seeding the held-out gold tests** to the agent during solve (any form of
  surfacing `test_patch` / the restored test files / a hidden `/tests` dir) — that is test
  leakage and invalidates the score.
- **Test-outcome manipulation** — config/hooks that rewrite test results to "passed" before the
  grader sees them.
- **Hardcoding return values for the exact hidden-test inputs.**
- **Reward-hacking / monkey-patching the grader** (stack introspection, operator overloading,
  patching the evaluator) instead of solving the task.

### The operator rule (the lesson that motivated this section)

When improving Pro results, ask: *does this change make the AGENT better (tools, prompt
discipline, grounding, reflection), or does it hand the agent the answer / the held-out
oracle?* The former is legitimate scaffold engineering; the latter is cheating. A finding from
forensics that "the agent failed because it could not see the gold tests" is NOT an argument to
show it the gold tests — it is an argument to make the agent reason from the contract.

`CONTAMINATION_AUDIT.md` records benchmark-defect false-negatives; it is **diagnostic-only and
never re-weights scores** — disclosure, not a replacement scorer.

### References (verified canonical sources)

- SWE-bench — Evaluation guide (patch-only prediction contract + Docker grader): <https://www.swebench.com/SWE-bench/guides/evaluation/>
- SWE-agent — Agent-Computer Interface paper (legitimate scaffold knobs; ~10.7-pt SWE-bench Lite gain from ACI alone): <https://arxiv.org/abs/2405.15793>
- SWE-bench issue #465 — the `git log --all` future-fix loophole (canonical report; maintainers report it fixed for Verified): <https://github.com/SWE-bench/SWE-bench/issues/465>
- NIST CAISI — Examples of cheating in CAISI's agent evaluations (git-history contamination): <https://www.nist.gov/caisi/cheating-ai-agent-evaluations/2-examples-cheating-caisis-agent-evaluations>
- NIST CAISI — Practices for detecting & preventing evaluation cheating: <https://www.nist.gov/caisi/cheating-ai-agent-evaluations/4-practices-detecting-and-preventing-evaluation-cheating>
- DebugML — Finding Widespread Cheating on Popular Agent Benchmarks: <https://debugml.github.io/cheating-agents/>
- Leaderboard/scaffold-practice analysis (validation, ranking, agent architectures across submissions): <https://arxiv.org/abs/2506.17208>
- SWE-bench Pro (paper): <https://arxiv.org/abs/2509.16941>
- Scale SEAL — SWE-bench Pro public leaderboard: <https://labs.scale.com/leaderboard/swe_bench_pro_public>
