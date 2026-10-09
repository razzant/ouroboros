# Ouroboros Terminal-Bench / Harbor Installed Adapter

## Short Summary

The current adapter runs **full Ouroboros inside each Terminal-Bench task
container**.

High-level flow:

1. Harbor creates the official Terminal-Bench task container.
2. The adapter uploads the current local Ouroboros `repo/` source into that
   container at `/opt/ouroboros-src`.
3. The adapter creates an isolated venv at `/opt/ouroboros-venv`.
4. The adapter installs Ouroboros from the uploaded source.
5. The adapter starts an in-container Ouroboros server/supervisor on
   `127.0.0.1:8765`.
6. The adapter submits the official Terminal-Bench instruction as an external
   workspace task, with `/app` or `/workspace` as the workspace root.
7. Ouroboros solves the task using its normal runtime/tools.
8. Harbor runs the official verifier.

This is intentionally **not** the old host-side terminal bridge. Ouroboros is
not asked to return one shell command per turn. It runs as normal inside the
task container.

## Why Installed Mode

The earlier adapter kept Ouroboros on the host and translated task state into a
JSON command loop. That made traces look artificially weak: Ouroboros saw a
terminal snapshot and had to return one shell command at a time.

The installed adapter evaluates Ouroboros more directly:

- each trial gets a fresh Ouroboros runtime;
- each trial gets a fresh `/logs/agent/ouroboros-data` data directory;
- the task workspace is passed as `workspace_root`;
- Ouroboros uses normal workspace tools and shell tools internally;
- Harbor still owns the task container and verifier.

## What Is Copied Into The Container

The adapter copies the current local source tree:

```text
/Users/anton/Ouroboros/repo -> /opt/ouroboros-src
```

It deliberately excludes local runtime/state noise:

```text
.git
.venv
data
data_evaluated
__pycache__
.pytest_cache
.ruff_cache
build
dist
node_modules
```

So the benchmark container gets current code, but not the operator's main
Ouroboros memory, logs, task results, or chat history.

The host-side adapter writes `source-provenance.json` in the Harbor agent log
directory before upload. It records source commit/version, dirty-state counts,
and hashes; it does not store full diffs or secrets. Publishable runs should use
a clean source tree or preserve this provenance beside the Harbor output.

## Runtime State In The Container

Each trial uses:

```text
OUROBOROS_REPO_DIR=/opt/ouroboros-src
OUROBOROS_DATA_DIR=/logs/agent/ouroboros-data
OUROBOROS_SETTINGS_PATH=/logs/agent/ouroboros-data/settings.json
OUROBOROS_RUNTIME_MODE=pro
OUROBOROS_REVIEW_ENFORCEMENT=blocking
OUROBOROS_TASK_REVIEW_MODE=required
OUROBOROS_SAFETY_MODE=light
OUROBOROS_MAX_WORKERS=4
OUROBOROS_MODEL_LIGHT=google/gemini-3.5-flash
OUROBOROS_WORKER_START_METHOD=spawn
```

This means:

```text
1 benchmark task = 1 fresh in-container Ouroboros = 1 unique ouroboros-data folder
```

The host `/Users/anton/Ouroboros/data` is not copied into the container.

## Provider Secret Boundary

Installed-container mode does not inject long-lived provider credentials into
Terminal-Bench task containers by default. If host settings or environment
contain provider keys, the adapter fails closed with a clear error instead of
starting a container that can expose those keys to in-container shell tools.

The intended durable solution is a reviewed host-mediated LLM bridge with scoped
task credentials. For trusted local smoke runs only, an operator may set:

```bash
OUROBOROS_BENCH_ALLOW_CONTAINER_SECRETS=1
```

Do not use that opt-in for publishable benchmark runs unless the task container,
logs, and output root are under operator control and the risk is explicitly
accepted.

## Task Instruction Integrity

The adapter passes the official Terminal-Bench instruction **verbatim**, then appends
exactly one harness-authored paragraph — an anti-lookup integrity clause
(`harbor_installed_agent.py`, in `run()`):

```
IMPORTANT - integrity: do not fetch this benchmark's task definitions, solutions,
tests, or reference materials from source repositories or mirrors on the internet.
Solve the task using the environment you are given; downloading general-purpose
software or data that the task itself requires is fine.
```

Nothing else is added: no task-specific hints, no solution guidance, no reordering or
rewording of the official text. The clause is a *restriction* on the agent, not help —
it forbids an avenue (fetching this benchmark's own task definitions from GitHub) that a
previous run was observed to use, and it cannot make an unsolved task solvable.

This paragraph is part of the measured configuration and MUST be disclosed in any
leaderboard submission: the exact text is reproduced above and lands verbatim in every
trial's `agent/instruction.txt`, so a reviewer can diff it against the official task.
An earlier version of this section claimed the instruction was passed "unchanged" and
that the adapter "does not prepend harness notes". That was false while the clause was
being appended, and a public artefact contradicting our own traces reads as concealment
rather than as an editing slip — hence the correction.

The only technical wrapper is the API request metadata and `workspace_root`.
For reliability it also passes:

```json
{
  "service_teardown": "keep",
  "timeout_sec": "<Harbor agent timeout when provided>"
}
```

`service_teardown=keep` prevents services started by Ouroboros from being
killed before Harbor's verifier connects to them. Harbor still owns the task
container and final cleanup boundary.

## Workspace Resolution

Most Terminal-Bench tasks use `/app`.

Some images use `/workspace`. The adapter resolves this before starting the
Ouroboros task:

1. use `/app` if it exists;
2. otherwise use `/workspace` if it exists;
3. otherwise create `/app`.

The selected path is passed as external workspace root.

## Lifecycle

Harbor calls:

```python
await agent.setup(environment)
await agent.run(instruction, environment, context)
```

`setup()` is inherited from Harbor's `BaseInstalledAgent`; it calls our
`install()`.

`install()`:

1. uploads clean Ouroboros source into `/opt/ouroboros-src`;
2. installs system basics (`git`, `curl`, `bash`, Python/venv support);
3. if the system Python is older than 3.10, installs Python 3.12 with `uv`;
4. creates `/opt/ouroboros-venv`;
5. installs requirements and editable Ouroboros.

`run()`:

1. uploads the task instruction to `/logs/agent/instruction.txt`;
2. checks configured provider/network reachability;
3. resolves `/app` vs `/workspace`;
4. ensures the workspace is a git worktree root;
5. starts in-container Ouroboros server;
6. creates an Ouroboros task through `/api/tasks`;
7. polls `/api/tasks/<task_id>` until a final status;
8. saves task result and trace files;
9. by default leaves the in-container server running until Harbor finishes the
   verifier/cleanup boundary, so `service_teardown=keep` services remain
   reachable for hidden verifiers; set `leave_server_running_for_verifier=false`
   only for local debugging where no post-run verifier needs live services.

## Why Direct API Polling

The adapter originally used:

```bash
ouroboros run --jsonl ...
```

That was fragile because the CLI stream could hang or get cancelled while the
internal task already had a final state.

The current adapter uses direct API lifecycle:

```text
POST /api/tasks
GET /api/tasks/<task_id>
POST /api/tasks/<task_id>/cancel
```

This gives the adapter a task id immediately, lets it capture task state on
timeout/cancellation, and avoids depending on an SSE/CLI stream.

## Timeout Semantics

The adapter does **not** set an internal task timeout by default:

```python
task_timeout_sec = None
```

That means Harbor controls agent execution timeout from the task config. When
Harbor provides that timeout to the adapter (`task_timeout_sec` agent-kwarg), it
is forwarded to Ouroboros so the agent sees deadline milestones without changing
official limits:

```text
task.toml [agent].timeout_sec
```

Honesty note: Harbor's `AgentContext` does not reliably pass `[agent].timeout_sec`
to installed agents, but the adapter now has a second legitimate fallback:
`_resolve_task_timeout_from_dataset` reads the public cached `task.toml` for the
current task package and forwards that official timeout to Ouroboros. Since v6.79.0
that lookup is DATASET-AWARE: the cache layout is
`~/.cache/harbor/tasks/packages/<org>/<name>/<digest>/task.toml` and the org is not a
constant (`terminal-bench/`, `gaia/`, `scale-ai/` coexist there), so the dataset identity
is threaded from the job config (`dataset` agent-kwarg) and the exact `<org>/<name>`
subtree is resolved first. A name-only fallback covers multi-org datasets but REFUSES when
two orgs cache the same task name, so a same-named task from another dataset can never
hand the agent a foreign cap — the run is deadline-blind instead. See
`METHODOLOGY.md` § "Dataset identity". Therefore
deadline milestones and deadline-derived `run_command` caps are usually active on
leaderboard-shaped cached runs; they are inert only when neither Harbor context nor
the public task cache exposes a timeout. Passing a synthetic `task_timeout_sec`
yourself remains a local experiment only; never inflate a task timeout for a
submission.

Setup and environment timeouts are separate:

- environment build/start: Harbor environment timeout;
- agent setup: Harbor setup timeout;
- agent execution: task `[agent].timeout_sec`;
- verifier: task `[verifier].timeout_sec`.

> ⚠️ **LEADERBOARD-DISQUALIFYING — do NOT use for a submission.** Setting
> `--environment-build-timeout-multiplier` or `--agent-setup-timeout-multiplier`
> to anything other than the default (null/1.0) makes the run **non-submittable**.
> The official Harbor leaderboard validator
> (`harbor/leaderboard/static_validation.py::_check_no_job_overrides`,
> verified in harbor 0.17.1 and upstream
> <https://github.com/harbor-framework/harbor>) rejects ANY non-null
> `agent_setup_timeout_multiplier` / `environment_build_timeout_multiplier`
> ("must not be set"), and **10/10 sampled real accepted submissions leave both
> `null`** (HF repo, see "Leaderboard Validity Rules" below). These are LOCAL-ONLY
> debug knobs. `run_tb.py` already guards them behind `--allow-setup-build-multipliers`
> and grades any such run `local_low_k`.
>
> **The faithful way to survive heavy/slow builds is NOT a multiplier — it is a
> pre-built/pinned image** (`environment.force_build=false`, which 8/10 real
> accepted submissions use). Pre-build the task images once and reuse them so the
> default 1.0 build/setup timeouts are never hit.

For heavy Docker builds **in a LOCAL (non-submission) run only**, you may use:

```bash
--environment-build-timeout-multiplier 4   # LOCAL ONLY — disqualifies a submission
--agent-setup-timeout-multiplier 4         # LOCAL ONLY — disqualifies a submission
```

## Leaderboard Validity Rules (verified 2026-07-07 against primary sources)

Triangulated from the OFFICIAL Harbor validator code (the same one Harbor Hub
runs on submit), the published submission docs, and real accepted submissions.
**A run is leaderboard-valid ONLY if ALL of these hold; otherwise it is LOCAL-only.**

| Rule | Requirement | Source |
|---|---|---|
| Trials per task `k` | **≥ 5** (`MIN_TRIALS_PER_TASK = 5`) | validator code; HF README; real submission path `…-k5-…` |
| Task `timeout_multiplier` | **== 1.0** | validator `_check_no_job_overrides`; timeouts post |
| `agent_timeout_multiplier`, `verifier_timeout_multiplier` | **must be null** | validator |
| **`agent_setup_timeout_multiplier`** | **must be null** (NOT a multiplier ≠1) | validator rejects any non-null; **10/10 accepted submissions = null** |
| **`environment_build_timeout_multiplier`** | **must be null** | validator rejects any non-null; **10/10 accepted submissions = null** |
| Resource overrides (`override_cpus/memory_mb/storage_mb/gpus`, `*.override_timeout_sec`) | **must be unset** | validator |
| Container network access | **ALLOWED** by default (task/package setup and in-container commands may use the network unless a task restricts it) | TB2.1 paper + Harbor task config |
| Ouroboros **agent-web tools** (`--allow-agent-web`) | **LOCAL-only / disclose** — static validation does not reject it, but enabling first-class web/search/browser tools raises reward-hacking exposure. Keep OFF for leaderboard-faithful runs unless the submission rules explicitly allow that exact scaffold | integrity update; `run_tb.py` self-stamps web-on runs non-faithful |
| Pre-built / pinned images (`environment.force_build=false`) | **ALLOWED & STANDARD** (TB2.1 reproducibility design) | **8/10 accepted submissions use `force_build:false`** |
| Host-side `colima` resources, `--n-concurrent` | **ALLOWED** (not job-config overrides) | not in validator |

Primary sources (read these before any submission-grade run):
- Official validator code: <https://github.com/harbor-framework/harbor> →
  `src/harbor/leaderboard/static_validation.py` (`MIN_TRIALS_PER_TASK`,
  `_check_no_job_overrides`, `_check_passing_trial_trajectories`). Installed
  locally as harbor `0.17.1`.
- **Submission process (PR-based, verified 2026-07-14):** submissions go
  through GitHub PRs in <https://github.com/harbor-framework/terminal-bench-2-1>
  (`leaderboard/SUBMIT.md` is the SSOT) — see "Submitting to the leaderboard"
  below. The old `harbor leaderboard submit` CLI was REMOVED (harbor PR #2230)
  and now fails server-side with `column leaderboard.slug does not exist`;
  `harbor hub leaderboard` is owner-only management, not a submission path.
- LEGACY: the HF dataset
  <https://huggingface.co/datasets/harborframework/terminal-bench-2-leaderboard>
  is the FROZEN 2.0 PR-based archive (last merged 2026-05-15). It is still
  useful to browse real accepted configs, but it is NOT the 2.1 submission
  channel — do not wait for it to "open".
- Reward-hacking judge + web policy:
  <https://www.tbench.ai/news/leaderboard-integrity-update>
- Timeout policy (task timeout must not be changed):
  <https://www.tbench.ai/news/leaderboard-integrity-and-timeouts>
- Run docs: <https://www.tbench.ai/docs/run-terminal-bench-2-1> ;
  leaderboard: <https://www.tbench.ai/leaderboard/terminal-bench/2.1> ;
  status/news: <https://www.tbench.ai/news>. Beware: as of 2026-07-07 the
  tbench.ai docs still say "submission process coming soon" — that page is
  STALE; the harborframework.com doc above is authoritative (this stale page
  already caused two false "submissions are closed" conclusions).

## Submitting to the leaderboard (GitHub PR flow, verified 2026-07-14)

The submission channel is the dataset repo
<https://github.com/harbor-framework/terminal-bench-2-1> (`leaderboard/SUBMIT.md`
= SSOT). Pipeline: public Hub job → `lb submit` opens a PR → CI static analysis
→ auto-promotion into a bot-owned PR (your PR gets CLOSED — this is normal) →
maintainer runs `/judge` (LLM reward-hacking review over every passing trial's
trajectory) → `/apply` → merge → the row lands on the leaderboard automatically.

```bash
harbor upload <job_dir> --public          # prints the job UUID ("View at …")
git clone https://github.com/<your-fork>/terminal-bench-2-1 && cd terminal-bench-2-1/leaderboard
# pre-fill src/leaderboard/display-names.json for your agent/model (headless
# `lb metadata` fails without a TTY on unknown names)
uv run lb submit <JOB_UUID>               # filter → metadata → open-prs
```

From a fork, `lb open-prs` pushes the branch but opens the PR without the fork
prefix — open it manually: `gh pr create --repo harbor-framework/terminal-bench-2-1
--head <you>:submission/<name> …`. The PR author can re-trigger CI with `/check`.

**CRITICAL — the agent must be launched with a NAMED job config.** Static
analysis matches `config.agents[].name` (job config) against the `agent_name`
trials report (the adapter class `name()`), and the trial join also uses that
name. A run launched with bare `--agent-import-path` records
`agents[0].name = null` and can NEVER pass ("no matching agent in job config",
terminal-bench-2-1#121 — four of our submissions were gated by exactly this).
`run_tb.py` therefore generates `agent_job_config.json` (agents[].name =
"Ouroboros Installed" + import_path + kwargs) and launches via
`harbor run -c <config>`; the job config on the Hub is immutable, so this is
unfixable after upload. Optional: `OUROBOROS_EFFORT_TASK` at launch is recorded
as `kwargs.reasoning_effort` (the submission's effort label) and forwarded back
into the container, so the declared effort equals the effective one.

Hard-won facts (verified against harbor 0.18.0 + leaderboard CI source):

- **Every REWARDED trial MUST have a trajectory** (ATIF, schema `ATIF-v1.7`):
  the leaderboard CI check "Rewarded trials have trajectories" reads the
  trial row's `trajectory_path` (the direct `trajectory.json` upload), and
  `/judge` reviews those trajectories. Runs from adapters that emit it live
  are fine; older runs can be backfilled with
  `build_atif_trajectories.py --job-dir <job> --validate` BEFORE uploading.
- **Trajectories must exist BEFORE the first `harbor upload`**: re-uploads
  skip trials that already exist server-side, so a trajectory added later is
  never attached — and if the direct trajectory PUT fails transiently the
  uploader silently degrades to archive-only (`trajectory_path = NULL`,
  unrepairable client-side: no update RPC, storage PUT blocked by RLS; this
  cost one Grok trial a red check). Verify post-upload:
  every rewarded trial on the Hub must have a non-null `trajectory_path`.
- **Submitted jobs become PUBLIC** so reviewers can inspect trials. If the
  run used `OUROBOROS_BENCH_ALLOW_CONTAINER_SECRETS=1`, every trial carries
  live provider keys in `agent/ouroboros-data/settings.json`. ALWAYS run
  `scrub_submission_secrets.py --root <job_copy> --secrets-from …` on a COPY
  of the job dir and verify 0 leftovers before uploading. If the run used
  `--agent-env`/`--verifier-env`, ALSO pass each pair as
  `--env-passthrough NAME=VALUE`: harbor writes those values into its own
  `config.json`/`lock.json`/`result.json` under the job dir (verbatim when the
  NAME does not look sensitive, otherwise as a partial `abcd****xyz` form), and
  they are in no `--secrets-from` source. An unsweepable value aborts the scrub
  instead of yielding a maybe-scrubbed tree.
- Auth is GitHub-OAuth on Harbor Hub; headless options: `HARBOR_API_KEY`
  env (API key from the Hub UI, cleanest for servers) or
  `harbor auth login --no-browser` + `--callback-url`.
- Multi-job submissions are supported (`lb submit <link> <link> …` /
  `source_jobs` lists them all): task coverage and ≥5 trials/task are
  evaluated over ALL jobs together — the sanctioned way to attach a rerun of
  failed/infra trials. Errored trials count as reward 0, never excluded.
- Task-version checking is content-hash based (per-trial `task.ref` sha256 vs
  the canonical dataset digests) and runs in CI static analysis.
- After static analysis + promotion the maintainer runs **/judge** (an LLM
  reads every passing trial's trajectory; harness-cheating findings
  invalidate the submission, reward-hacking findings zero the flagged trials
  at `/apply`) — emit honest, information-rich trajectories, never stubs.
  Large trial artifacts (>~350 MB single POST) used to fail upload with 403
  "exp claim" — fixed by routing big archives through the TUS path
  (harbor#2318); keep an eye on per-trial upload results.

### Cost reality (don't burn money on non-faithful full runs)
A FULL run is expensive: gpt-5.5-high on TB2.1 (89 tasks) costs **~$1.5/trial
average** (median ~$1.2, worst single task ~$12). So **k=3 ≈ $330–420**, and a
faithful **k=5 ≈ $550–700**. Per-trial wall-clock is ~75% LLM solving (~14 min at
n-concurrent 3). **Before launching a full run, confirm the config is leaderboard-valid
(table above) — a wrong knob (e.g. a setup/build multiplier, k<5) means the whole
spend is non-submittable and must be re-run.** Cost is in each trial's
`agent/ouroboros-run-summary.json` → `cost_usd` (sum across trials for the run total;
note deleted/retried trial dirs drop their cost record, so the on-disk sum is a lower bound).

### Hard-won errors / gotchas (so we don't repeat them)
- **×4 setup/build multipliers are NOT faithful.** The old advice in "Timeout
  Semantics" (and example commands) used them; they DISQUALIFY a submission. Use
  **pre-built images** instead to survive slow/heavy builds at the default 1.0 timeouts.
- **Distinguish container network from Ouroboros web tools.** Terminal-Bench tasks
  normally allow container-level network access for package installs and task work, but
  `--allow-agent-web` exposes Ouroboros's first-class web/search/browser tools. That is
  a scaffold change with reward-hacking risk (TB site/GitHub/online solution lookups),
  so keep it OFF for leaderboard-faithful runs and disclose it for local experiments.
  For example, `mteb-leaderboard` must reach its answer via official sources, not a
  3rd-party TB "explorer" that leaks the reference + canary.
- **Container-secret env var is `OUROBOROS_BENCH_ALLOW_CONTAINER_SECRETS=1`** (full
  name; the bare `ALLOW_CONTAINER_SECRETS` silently fails every task).
- **`run_tb.py` flag names differ from harbor's:** `--setup-timeout-multiplier` /
  `--build-timeout-multiplier` (run_tb) map to
  `--agent-setup-timeout-multiplier` / `--environment-build-timeout-multiplier` (harbor).
- **No-resume fragility of `run_tb.py`** (fresh job + `--force-build` each call) vs the
  robust path: **`harbor job resume -p <jobdir> [-f <ErrorType>]`** continues an existing
  job (keeps completed trials, re-runs pending + the `-f`-removed errored-artifact trials);
  wrap it in a retry loop so a transient SSL/DNS blip just re-resumes instead of restarting.
- **install-timeout on slow mirrors** (`RuntimeError: Command timed out after 1200s`) and
  AgentSetupTimeout are infra, not capability — pre-built images remove this failure class.
- **Pausing via SIGSTOP + sleep** blows in-flight trials' wall-clock deadlines →
  `deadline_local` reward-0 artifacts on resume; reclassify (reason_code + pause window)
  and re-run those trials before scoring. Don't count infra/pause artifacts as genuine fails.

## Common Commands

### Publishable Terminal-Bench 2.1 run

Use `run_tb.py` for leaderboard-shaped runs. It enforces the public
methodology constraints we care about locally: `k >= 5`,
`timeout_multiplier == 1.0`, no resource overrides, and a generated
`metadata.yaml` under the submission tree.

```bash
PYTHONPATH=/Users/anton/Ouroboros/repo \
python devtools/benchmarks/terminal_bench/run_tb.py \
  --model openai/gpt-5.5 \
  --k 5 \
  --n-concurrent 1 \
  --run-root /Users/anton/Ouroboros/bench_runs/terminal_bench/tb21_gpt55 \
  --submission-root /Users/anton/Ouroboros/bench_runs/terminal_bench/submission \
  --execute
```

For a targeted smoke, add repeated `--task` filters before `--execute`, for
example `--task pypi-server --task hf-model-inference --task qemu-alpine-ssh`.

The launcher refuses to start from a dirty/unidentifiable seed checkout (the shared
`benchmark_run_manifest` gate, fail-closed since v6.75.0); `--allow-dirty-seed` records the
exception instead. Additional readiness flags (v6.79.0, all optional and all off by
default): `--base-job-config PATH` deep-merges an upstream Harbor JobConfig under our
`agents[]` block, `--agent-env KEY=VALUE` / `--verifier-env KEY=VALUE` forward harbor's own
`--ae`/`--ve` (values are redacted out of `harbor_command.txt`, stdout and the manifest,
which keeps NAMES only — but harbor itself PERSISTS those values into the job dir, so a
submission copy must additionally be scrubbed with `scrub_submission_secrets.py
--env-passthrough NAME=VALUE`; see METHODOLOGY "Agent/verifier env passthrough"), and
`--submission-subtree` overrides the submission path derived
from `--dataset`. Frontier-Bench runs on the pinned 0.18.0 (measured — see METHODOLOGY
"Frontier-Bench"); a dataset that needs a newer harbor flag runs from `~/ouro/venv-fb`
(0.20.0) via `--harbor-bin`, leaving `venv-tb` frozen.

### Terminal-Bench 2.1 smoke

Ledgered smoke runs should go through the wrapper so `run_manifest.json` and
the denominator-preserving `result_index.jsonl` are written beside the Harbor
official output. Its manifest derives Main/Light/fallback and the one API actor
from the exact `--model` CLI value (plus an explicit Light override), never from
ambient Heavy or a settings-template model:

```bash
PYTHONPATH=/Users/anton/Ouroboros/repo \
python devtools/benchmarks/terminal_bench/run_harbor_smoke.py \
  --run-root /Users/anton/Ouroboros/bench_runs/terminal_bench/smoke \
  --task terminal-bench/regex-log \
  --model openai/gpt-5.5 \
  --n-concurrent 1 \
  --execute
```

Raw Harbor commands are useful for local debugging of the installed agent, but
they do not write the Ouroboros denominator ledger unless wrapped by
`run_harbor_smoke.py`. **They are also NOT submission-valid**: bare
`--agent-import-path` records `agents[0].name = null` in the job config, which
the TB2.1 leaderboard CI can never match (see "Submitting to the leaderboard").
Use `run_tb.py` (which launches via a named job config) for anything that might
be submitted.

```bash
PYTHONPATH=/Users/anton/Ouroboros/repo \
harbor run \
  --dataset terminal-bench/terminal-bench-2-1 \
  --include-task-name terminal-bench/regex-log \
  --agent-import-path devtools.benchmarks.terminal_bench.harbor_installed_agent:OuroborosTerminalBenchAgent \
  --model ouroboros-gpt-5.5-tb21-smoke \
  --agent-kwarg ouroboros_model=openai/gpt-5.5 \
  --agent-kwarg install_timeout_sec=1200 \
  --agent-kwarg server_start_timeout_sec=240 \
  --agent-setup-timeout-multiplier 4 \
  --n-concurrent 1 \
  --n-tasks 1 \
  --yes \
  --force-build
```

### Full cached Terminal-Bench 2.0-style dataset

Debug-only raw Harbor form; for publishable ledgered runs, mirror these options
through `run_harbor_smoke.py` or write an explicit wrapper that emits
`run_manifest.json` and `result_index.jsonl`.

```bash
PYTHONPATH=/Users/anton/Ouroboros/repo \
harbor run \
  --path /Users/anton/Ouroboros/data/harbor_local_datasets/terminal_bench_full_cached_89 \
  --agent-import-path devtools.benchmarks.terminal_bench.harbor_installed_agent:OuroborosTerminalBenchAgent \
  --model ouroboros-gpt-5.5-full \
  --agent-kwarg ouroboros_model=openai/gpt-5.5 \
  --agent-kwarg install_timeout_sec=1200 \
  --agent-kwarg server_start_timeout_sec=240 \
  --agent-setup-timeout-multiplier 4 \
  --n-concurrent 1 \
  --yes \
  --force-build
```

### Full Terminal-Bench 2.1

Debug-only raw Harbor form; it preserves Harbor's official output but not the
Ouroboros denominator ledger.

```bash
PYTHONPATH=/Users/anton/Ouroboros/repo \
harbor run \
  --dataset terminal-bench/terminal-bench-2-1 \
  --agent-import-path devtools.benchmarks.terminal_bench.harbor_installed_agent:OuroborosTerminalBenchAgent \
  --model ouroboros-gpt-5.5-tb21-full \
  --agent-kwarg ouroboros_model=openai/gpt-5.5 \
  --agent-kwarg install_timeout_sec=1200 \
  --agent-kwarg server_start_timeout_sec=240 \
  --agent-setup-timeout-multiplier 4 \
  --environment-build-timeout-multiplier 4 \
  --n-concurrent 1 \
  --yes \
  --force-build
```

## Model Selection

Harbor's `--model` is metadata for the Harbor result.

The actual Ouroboros model is passed via:

```bash
--agent-kwarg ouroboros_model=<provider/model>
```

Examples:

```bash
--agent-kwarg ouroboros_model=openai/gpt-5.5
--agent-kwarg ouroboros_model=google/gemini-3.5-flash
--agent-kwarg ouroboros_model=anthropic/claude-opus-4-7
```

The adapter sets:

```text
OUROBOROS_MODEL
OUROBOROS_SUBAGENTS = one api_model row on OUROBOROS_MODEL
```

inside the container. The explicit actor row prevents Available-subagent defaults
from introducing a Light scout, second family, or session-backed delegate into a
fixed-model trial. `OUROBOROS_MODEL_LIGHT` defaults to
`google/gemini-3.5-flash` and can be overridden with
`--agent-kwarg ouroboros_light_model=<provider/model>` or `run_tb.py
--light-model ...`. This avoids accidentally running safety checks and
lightweight JSON decisions on the expensive measured model.

### Why `--all-model` pins the review slots too

`run_tb.py --all-model` pins the review pool (ONE packet review seat on the solve model,
written as a catalog row of `OUROBOROS_SUBAGENTS`, at low effort) and the Light lane to
the solve model. This is intentional and must stay:
a TB run claims a SINGLE-MODEL measurement, so the acceptance-review content —
which feeds improvement passes back into the answer — must come from the same
model. Substituting a stronger/different reviewer would smuggle a second
reasoning model into the scaffold and invalidate the single-model claim; the
lone low-effort reviewer slot keeps review ON (part of the measured harness)
without reviewer diversity. `single_reviewer_no_diversity` stays loud in logs
by design. The Light pin also covers post-task synthesis and memory
consolidation, so late model work cannot silently introduce another model into
the physical trial.

### Scaffold defaults (v6.55.0)

The adapter template pins, and the methodology discloses:

- `OUROBOROS_RUNTIME_MODE=pro` — the container is a disposable jail with a
  fresh repo copy; pro unlocks the file/self-modification surface the bench
  legitimately measures.
- `OUROBOROS_MAX_WORKERS=4` (was 2) — same-model subagent slots for
  decomposition within one trial; the root agent occupies one lane. Higher
  values blow container memory (each worker is a full Python process).
- `OUROBOROS_SAFETY_MODE=light` — the jail is isolated; the LLM safety pass
  was 34% of all LLM calls in the k=5 run while the deterministic guards do
  the actual protecting. Light keeps the LLM check for integration tools only.
- `claude_code_edit` disabled in every trial — benches measure the
  single-model Ouroboros harness; an external agent-session delegate is a
  separate experiment.
- `_DEADLINE_SAFETY_SEC=105` (was 30) — measured finalization overhead plus a
  provider-recovery margin, so trials finalize before Harbor's hard deadline
  instead of losing a finished answer (gpt2-codegolf overran by 26.5s at 30).

## Infra-Failure Semantics

OpenRouter credit exhaustion used to produce quiet zero-reward tails. The
adapter now:

- runs a host-side OpenRouter credit preflight when a key is configured
  (`OUROBOROS_BENCH_OPENROUTER_MIN_CREDIT_USD`, default `$5`);
- treats `llm_api_error` / `infra_failed` as adapter errors rather than
  ordinary semantic failures;
- writes `openrouter-credit-preflight.json` beside the agent logs.

## Trace Locations

For each Harbor trial:

```text
<trial>/agent/ouroboros-data/
```

contains the fresh in-container Ouroboros data directory.

Useful files:

```text
<trial>/agent/ouroboros-data/logs/events.jsonl
<trial>/agent/ouroboros-data/logs/progress.jsonl
<trial>/agent/ouroboros-data/logs/supervisor.jsonl
<trial>/agent/ouroboros-data/state/headless_tasks/<task_id>/data/logs/tools.jsonl
<trial>/agent/ouroboros-task-result.json
<trial>/agent/ouroboros-run.jsonl
<trial>/agent/ouroboros-run-summary.json
<trial>/verifier/test-stdout.txt
<trial>/verifier/reward.txt
```

Heavy files usually come from:

```text
<trial>/agent/ouroboros-data/task_results/artifacts/<task_id>/workspace.patch
<trial>/agent/ouroboros-data/task_results/artifacts/<task_id>/workspace_patch.json
```

Those can be omitted when creating logs-only bundles.

## Known Infrastructure Notes

- Old task images with Python 3.9 require adapter-installed Python 3.12 via
  `uv`; this is handled automatically.
- Some task Docker builds need more than 600 seconds; use
  `--environment-build-timeout-multiplier`.
- Some tasks still hit Harbor `AgentTimeoutError`; verifier can still produce a
  reward if the workspace has enough final state.
- `RuntimeError` from the adapter should not be used for ordinary Ouroboros
  `status=failed`; the adapter records task status and returns control so Harbor
  can run the verifier.

## Files To Share With Developers

Minimum:

```text
repo/devtools/benchmarks/terminal_bench/harbor_installed_agent.py
```

Recommended:

```text
repo/devtools/benchmarks/terminal_bench/README.md
```

Useful example result:

```text
data/harbor_jobs/ouroboros_v650_tb21_smoke_gpt55/2026-05-29__00-39-23/result.json
```
