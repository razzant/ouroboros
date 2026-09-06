# Terminal-Bench 2.1 — Methodology

How Ouroboros runs Terminal-Bench 2.1 (harbor) and what a published number means.
Companion to `README.md` (mechanics) — this file is the disclosure/validity SSOT
for TB runs, in the same spirit as `../swe_bench_pro/METHODOLOGY.md`.

## CLEAN SEED PRECONDITION (MANDATORY)

**A SUBMITTABLE TB RUN MUST START FROM A CLEAN SEED WORKTREE:
`git status --porcelain` EMPTY, `git describe --dirty` WITHOUT A `-dirty`
SUFFIX. A DIRTY SEED POISONS THE RUN MANIFEST (`...-dirty`), MAKES PROVENANCE
NON-REPRODUCIBLE, AND DISQUALIFIES THE RUN FROM LEADERBOARD SUBMISSION.
COMMIT ADAPTER EDITS (WIP COMMIT RECORDED IN RUN NOTES IS FINE) AND REMOVE
STRAY FILES BEFORE LAUNCH; IF DIRT IS FOUND AFTER LAUNCH, ESCALATE TO THE
OWNER IMMEDIATELY.**

## Protocol

- Official harbor harness and task images; `run_tb.py` launches the installed
  full-Ouroboros agent (`harbor_installed_agent.py`) — the agent installs its own
  venv prefix inside each task container (system Python, `pip install -e`, and
  since v6.56.0 `imageio-ffmpeg` in the SAME agent prefix so `extract_video_frames`
  works without touching task-owned packages; a failed install degrades to the
  typed `EXTRACT_VIDEO_FRAMES_UNAVAILABLE` + cv2 workaround hint).
- Agent-setup phase installs AGENT dependencies only — never task dependencies,
  never edits to task files; solving starts only after install completes.
- Harbor owns task timeouts; Ouroboros deadline milestones are inert unless the
  task carries `deadline_at`. The in-container finalization margin is
  `_DEADLINE_SAFETY_SEC=105` (measured overhead, v6.55.0).
- k trials per task follow the leaderboard's k; `disclosure_ledger.json`
  (schema `tb_disclosure_ledger.v1`) records the reward distribution,
  timeout/rate-limit/provider-failure histograms, per-task pass rate,
  concurrency, and the multiplier/gating flags actually used, so each run's
  leaderboard validity is auditable after the fact.

## Dataset identity (v6.79.0 — matters for any non-TB2.1 Harbor dataset)

`run_tb.py` is dataset-parametric, so the dataset identity is now threaded instead of
assumed in two places that used to hardcode TB2.1:

- **Per-task wall-clock cap.** The adapter reads `[agent].timeout_sec` from the cached
  `task.toml` at `~/.cache/harbor/tasks/packages/<org>/<name>/<digest>/task.toml`
  (harbor's own `PackageTaskId.get_local_path`, checked against harbor 0.18 and 0.20).
  The org is NOT a constant — that cache already holds `terminal-bench/`, `gaia/` and
  `scale-ai/` side by side, and a multi-org dataset such as Harbor-Index ships tasks from
  several orgs at once — so the previous hardcoded `terminal-bench` literal silently made
  every other dataset run deadline-blind. The dataset (`<org>/<name>`) now travels through
  the job config into the adapter, which resolves the exact `<org>/<name>` subtree.
  A configured org is AUTHORITATIVE: if that org's cache holds no such task, no cap is used
  at all (deadline-blind, as before) — there is deliberately no fallback to another org's
  cached entry, because the borrowed field is the wall-clock cap and borrowing it runs the
  task under a different benchmark's deadline (`frontier-bench/frontier-bench` verifier caps
  are 600s against terminal-bench-2-1's 3600s, and both are routinely cached side by side).
  Harbor's `AgentContext` carries no task identity, so a dataset given with NO org at all
  still falls back to a name-only search — and that search REFUSES on ambiguity: if two orgs
  cache the same task name, no cap is used rather than a foreign dataset's cap.
- **Submission subtree.** The leaderboard tree was hardcoded as
  `submissions/terminal-bench/2.1/…`, so a run on any other dataset would have been filed
  under the TB2.1 tree. It is now derived from the dataset name
  (`<org>/<family>-<major>-<minor>` → `<family>/<major>.<minor>`), which keeps the TB2.1
  output byte-identical. **Disclosure:** the derivation only guarantees TB2.1's layout. The
  exact expected path for any OTHER leaderboard must be read from that leaderboard's own
  `SUBMIT.md` at submission time (submission mechanics are the fastest-rotting fact in this
  area — `harbor leaderboard submit` was removed upstream within weeks); use
  `--submission-subtree` when it differs. For a local smoke this path is cosmetic.

## Agent/verifier env passthrough and web tools (v6.79.0)

- `--agent-env`/`--verifier-env` forward `KEY=VALUE` pairs to harbor's own `--ae`/`--ve`
  (present in harbor 0.18 and 0.20). These are deploy knobs like `--n-concurrent`, not
  leaderboard-config fields, so they do not affect `static_validation`.
- **Where those VALUES end up (corrected — the earlier "values never enter the run root"
  claim was false, and false on the Harbor-Index path where the judge key goes through
  `--ve` and the upload is public).** Two different writers:
  * *Ours*: no value enters any artifact this launcher writes — the manifest records
    `agent_env_keys`/`verifier_env_keys` (NAMES only) plus the typed
    `env_passthrough_persisted_by_harbor` fact, and `harbor_command.txt` and stdout print
    a redacted command.
  * *Harbor's*: harbor persists its own `JobConfig` INTO the job directory, i.e. inside the
    tree that gets uploaded — `harbor/job.py` writes
    `config.model_dump_json(indent=4, exclude_defaults=True)` to
    `<jobs-dir>/<job_name>/config.json` (one timestamp level below the `--jobs-dir` we
    pass), and the same env dicts are re-serialized into the job `lock.json` and every
    trial's `config.json`/`lock.json`/`result.json`. `--ae` lands in `agents[].env`, `--ve`
    in `verifier.env`. Harbor's `templatize_sensitive_env` (`harbor/utils/env.py`) filters
    them by variable NAME only. Measured against the installed 0.18.0: a value equal to
    `os.environ[NAME]` becomes `${NAME}` (no leak); a NAME matching
    `KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL|AUTH` is written as
    `value[:4] + "****" + value[-3:]` — a *partial* disclosure, seven characters of a live
    credential, not a redaction; **any other NAME (e.g. `MY_BEARER`) is written verbatim in
    cleartext.** Harbor's `exception.txt` and agent-written log dirs can also carry the
    resolved value.
  * Therefore the submission copy must be scrubbed BY VALUE:
    `scrub_submission_secrets.py --root <job_copy> --secrets-from … --env-passthrough
    NAME=VALUE` for every pair given to `--ae`/`--ve`. It sweeps the whole tree (harbor's
    config/lock/result files included, no filename special-case), sweeps harbor's partial
    form as its own needle, verifies zero leftovers independently, and REFUSES (exit 2,
    nothing modified) when a value cannot be swept safely — a maybe-scrubbed public upload
    is worse than no submission.
  * The sweep is not literal-only. Everything harbor persists an env value into is JSON, so a
    value containing a quote, a backslash, a control character or a non-ASCII character is on
    disk ESCAPED (`abc"12345678` is stored as `abc\"12345678`). Every needle is therefore
    expanded into the forms a JSON writer produces — derived by running `json.dumps` itself,
    both `ensure_ascii` settings, never a hand-kept escape table — and BOTH the replacement
    pass and the independent verify pass use that expanded set, so `verify_leftovers=0` cannot
    be reported while a secret survives in any persisted form.
- `--base-job-config` deep-merges an upstream JobConfig UNDER our block: every key we do
  not set survives verbatim, while `agents[]` — including `agents[0].name`, whose absence
  permanently invalidates a submission (terminal-bench-2-1#121) — always stays ours.
- **Agent web tools stay OFF by default** (`--disable-agent-web`, the reward-hacking guard)
  and `metadata.yaml` declares no `web_search` role while they are off. A dataset whose
  own rules permit web access is run with the explicit `--allow-agent-web` flag, which
  prints a loud non-leaderboard-faithful warning for TB2.1 and is recorded in the manifest
  and the disclosure ledger — never a silent default.
- **Every reviewer row the container can run is declared; the rest is disclosed (2026-09-02).**
  Task acceptance executes every triad row on its configured delivery (owner R2), so
  `metadata.yaml` declares the rows the container's panel actually runs — api packet rows and
  configured-subagent native inspection rows, by model id. An agent-session row structurally
  cannot run inside a Terminal-Bench task container: the image has no harness CLI/daemon, the
  forwarded-env allowlist carries no harness credentials, and the container secret policy
  forbids them. It is therefore never declared as a used model (a declared-but-never-run model
  would misrepresent the submission) and is carried as the typed disclosure
  `triad_rows_not_executable_in_container` (the rows' `harness[=model]` targets) in
  `run_manifest.json`, with the same list as a comment line in `metadata.yaml`; its acceptance
  seat degrades typed inside the container, so configure api/native rows for a TB run. Before
  this change only the panel's non-retrieving api rows ran and were declared, with the shipped
  defaults substituted when none existed; runs whose panel carries retrieving rows are not
  comparable on the acceptance axis with earlier runs, and a submission must be read against
  its own `metadata.yaml` and manifest. `metadata.yaml` cannot distinguish an api packet row
  from a configured-subagent native inspection row: both are declared under
  `commit_review_triad` by model id, and under the container's one-model roster a
  subagent-bound row resolves to the measured model and dedupes onto it
  (`agent+commit_review_triad`). The per-delivery record is the run manifest: on a fixed-model
  run `harness.fixed_model_actor.reviewer_slots` carries the projection's `slot_id`,
  `route{kind, target_id}` and `effort` per row — a fixed-model panel is always direct api
  rows, so no subagent binding exists there — and `extra.triad_rows_not_executable_in_container`
  carries the session rows. A plain `--model` run records only the latter; its api-vs-native
  split is recoverable from the forwarded host settings (`--settings-path`) only when the panel
  is persisted there — the container adapter resolves the environment first
  (`harbor_installed_agent.py`, `_container_env`), so a panel supplied only through the
  operator's environment leaves no durable per-delivery record on a plain `--model` run.
- **Harbor version:** the pinned TB2.1 bench venv is harbor **0.18.0** (`~/ouro/venv-tb`). 0.20.0
  is the current latest and is installed in a SEPARATE venv (`~/ouro/venv-fb`), reachable via
  `--harbor-bin` and leaving `venv-tb` frozen at 0.18.0 so published TB2.1 numbers keep their
  harness. Note `--base-job-config` is OURS, not a harbor flag — harbor only ever had
  `-c/--config`, in both versions. Since v6.79.0 the manifest and the disclosure ledger record
  `harbor_bin` / `harbor_version` / `harbor_env_effective`, so the harness and backend behind a
  number are read off the artifact instead of reconstructed from operator memory; an
  un-interrogable binary records `harbor_version: ""` (visibly unknown, never assumed).

## Frontier-Bench (readiness only — NO run has been scored)

Frontier-Bench is Terminal-Bench's declared successor (harbor + Laude Institute, data/research
contributions from Snorkel AI), and it is **a harbor dataset, not a new harness**. Consequences for
this launcher, all verified on this host rather than assumed:

- **Identity:** `frontier-bench/frontier-bench` (`FRONTIER_BENCH_DATASET` in `run_tb.py`), ref
  `latest`. Harbor 0.18.0 resolves and downloads it (74 tasks). `latest` is a **mutable tag row**,
  so a reproducible run MUST pin an immutable ref — `@v0.1.0`, `@<revision>` or
  `@sha256:<digest>`. Verified with a negative control on harbor 0.18.0: `@v0.1.0` resolves 74
  tasks, a bogus `@v9.9.9` exits non-zero. Published refs at the time of writing: `v0.0.1`,
  `v0.1.0` (== `latest`, revision 3).
- **Task shape is TB2.x-identical** (`task.toml`, `instruction.md`, `environment/Dockerfile`,
  `tests/test.sh`, `solution/solve.sh`), so the installed-agent adapter is compatible unchanged and
  the dataset-parametric wall-clock lookup works as-is: the cache subtree is
  `~/.cache/harbor/tasks/packages/frontier-bench/<task>/<digest>/task.toml`.
- **Backend: local docker is sufficient — a cloud sandbox is NOT required.** Upstream develops and
  runs its CI/leaderboard on Modal (`--env modal`), which is easy to misread as a hard requirement.
  Measured instead: the oracle solution of a real FB task (`bun-sourcemap-leak`) scored **reward
  1.0, 0 exceptions, 69 s** on harbor 0.18.0 against the local rootless docker daemon, through the
  same separate-environment verifier the benchmark scores with. Backend selection is now explicit
  via `--harbor-env` (harbor's own `-e/--env`) and is disclosed either way.
- **What local docker does NOT cover:** 4 of 74 tasks request `gpus = 1`
  (`exam-pdf-eval`, `fp8-rmsnorm-gemm`, `jax-speedrun-gpu`, `math-eval-grader`) and one of those
  also requests 1 TB of storage; 1 task (`medical-claims-processing`) declares MCP servers. Those 4
  REFUSE loudly rather than silently mis-scoring: harbor's docker backend advertises no GPU
  capability, so `_validate_gpu_support` raises before the agent starts (the backend contains no
  nvidia wiring at all, so this host's own H200s do not change it). 12 tasks use multi-container
  `docker-compose`, which local docker handles natively. A partial run MUST
  disclose which tasks were excluded and why — an unscored task is not a failed task.
- **Cost/time shape (the reason a "small" smoke is not cheap):** per-task agent timeouts are
  `[agent].timeout_sec` **median 7200 s, min 1800 s, max 28800 s** — an order above TB2.1 — and the
  expert-time estimates run 0.75–60 h. Task selection for a smoke must therefore be driven by these
  declared caps, not by task count.
- **Scoring and submission:** harbor scores locally (reward per trial, mean in the job
  `result.json`), so a smoke needs **no submission, no upload, and no Hub account**. Leaderboard
  submission mechanics for FB are NOT established here and must be read from upstream at submission
  time — the same fastest-rotting-fact rule that already applies to TB2.1.
- **Contamination canary:** every FB `task.toml` carries a `harbor-canary GUID` line. Treat it like
  task content: never quote it into a public artifact, a commit message, or a model prompt beyond
  the container it already lives in.

## Review-mode disclosure

The recorded best configuration used BLOCKING task-review. Campaign runs may
deliberately use `advisory` review (e.g. the v6.56.0 gpt/gemini rows) to trade
review latency for wall-clock throughput — that is a DISCLOSED deviation from
the record configuration and the run manifest carries the actual review mode;
numbers from advisory rows must not be presented as the blocking-config record.

Acceptance improvement passes are bounded by the owner's shared `OUROBOROS_REVIEW_MAX_CYCLES`
(default 2, `unlimited` available) since 2026-08-15 — including under `required`+`blocking`,
which had no local count cap before. This template carries no explicit value, so runs use the
default; an exhausted cap terminates honestly with the typed `review_cycles_exhausted` reason.
Runs from before that change are not comparable on this axis.

## Leaderboard submission

Since ~2026-07-08 the official TB2.1 submission channel is the GitHub PR flow
in <https://github.com/harbor-framework/terminal-bench-2-1>
(`leaderboard/SUBMIT.md` = SSOT): public Hub job → `lb submit` opens a PR →
CI static analysis → auto-promotion (bot re-owns the trials and the PR) →
maintainer `/judge` + `/apply` → merge = the leaderboard row. The previous
`harbor leaderboard submit` CLI was removed (harbor PR #2230) and the old HF
PR flow is a frozen 2.0-only archive. See README "Submitting to the
leaderboard" for the verified mechanics. Methodology-relevant consequences:

- **The run must be LAUNCHED with a named job config** (`harbor run -c` with
  `agents[].name` set to the adapter class name). Bare `--agent-import-path`
  records `agents[0].name = null` in the job config, which static analysis
  can never match against the trial-side agent name — the submission is
  permanently invalid and the uploaded job config cannot be fixed post hoc
  (terminal-bench-2-1#121). `run_tb.py` generates `agent_job_config.json`
  accordingly; `OUROBOROS_EFFORT_TASK` is recorded as the declared
  `reasoning_effort` of the submission key and forwarded into the container.
- CI requires an **ATIF trajectory for every rewarded trial** (the direct
  `trajectory.json` upload; the trial row must carry `trajectory_path`). The
  adapter emits it in-container at the end of each trial; pre-existing runs
  are backfilled with `build_atif_trajectories.py` BEFORE the first upload.
  Trajectories are derived verbatim from the recorded logs (tool calls,
  progress narration, final answer, token/cost totals) — no synthesis. The
  maintainers run an LLM judge over these trajectories (reward-hacking
  review; flagged trials are zeroed at `/apply`).
- Submitted jobs become **public**. Because campaign runs inject provider
  keys into task containers (disclosed limitation below), the submission copy
  of a job MUST pass `scrub_submission_secrets.py` (structural blanking of
  embedded settings + value sweep over the literal AND JSON-escaped forms +
  zero-leftover verification over that same expanded set)
  before upload. Scrubbing only removes credential values; results, traces,
  and disclosure artifacts are uploaded unmodified.
- Numbers submitted to the leaderboard must come from `leaderboard_valid`
  runs (k≥5, no overrides, agent-web off) under the review-mode disclosure
  rules above; advisory-mode rows must not be presented as the blocking
  record.

## Known limitations

- **Ouroboros data root visible to task shells (masking not yet applied).** The
  agent's own data root lives inside the task container and a task shell could
  in principle read it. The bwrap filesystem isolation precedent (GAIA, ef363ff)
  is the candidate fix; applying it to TB is deliberately DEFERRED to its own
  design review (v6.56.0 sprint decision) — TB campaign runs execute WITHOUT the
  mask and this is a disclosed known limitation, mitigated by trace audits
  (reward-hacking sweep over tools.jsonl / solve traces).
- **The OpenRouter key preflight is not a guarantee.** Since v6.79.0 it reads the
  authoritative `/api/v1/key` `limit_remaining` (the old `total_credits − total_usage`
  arithmetic lies on a nearly exhausted key), but a SHARED key can still report headroom a
  neighbour spends before the trial starts, and an uncapped key (`limit: null`) has no
  threshold at all. Only a real completion is proof.
- Infra failures (DNS, 429 storms, install timeouts) are classified
  infra-vs-genuine BEFORE scoring; k<5 partials are marked low_k and never
  compared against the leaderboard.
