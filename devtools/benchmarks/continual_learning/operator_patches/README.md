# Operator patches for the external CLB adapter (campaign v6.56.0)

The pinned Ouroboros adapter (`src/systems/ouroboros/`, handoff bundle v3,
reference commit 56764d6) predates v6.56.0 and hits three host/runtime
incompatibilities on a Linux host. These unified diffs port it; apply them on
top of the bundle copy inside the external continual-learning-bench checkout
(`patch -p0 < <file>` from the checkout root, adjusting paths):

1. `_launcher.v6560.patch` — write the declared `OUROBOROS_SAFETY_MODE=light`
   and the canonical `OUROBOROS_SUBAGENTS` bytes into isolated settings at
   creation. `IsolatedServer` strips inherited runtime env; this makes the
   benchmark's explicit one-actor profile authoritative there too. v6.56.0
   added an owner-guard that
   refuses env-side `full -> light` lowering at server boot
   (`_guard_safety_mode_lowering`), which killed every isolated server with
   uvicorn rc=3; the reference v6.52.2 run had no guard and was effectively
   `light`. Writing the knob into settings from birth keeps the guard intact
   (no lowering happens) and restores reference parity.
2. `clbench_step_shim.v6560.patch` — make the step-shim bind address
   overridable via `CLBENCH_SHIM_BIND` (default remains 127.0.0.1). On Linux,
   `--add-host=host.docker.internal:host-gateway` does NOT reach a
   host-loopback listener (empirically verified on both rootful and rootless
   daemons), so the docker-engine agent's submit_action silently never reached
   the shim: every question "completed" with `queries=0, reward=None`. Bind to
   the docker bridge IP (e.g. 172.17.0.1) for docker-engine runs.
3. `_docker_launcher.v6560.patch` — run the agent container with
   `--user <uid>:<gid>` (+ `HOME=/obo/data`). On a rootful daemon the
   container wrote root-owned files into the bind-mounted data root, and the
   host-side bridge died with PermissionError.

Also required for the docker engine: seed `skills/clbench_remote/` (from the
bundle's `bench-config/external-adapters/clbench_remote/`) into the dedicated
Ouroboros clone — the v6.56.0 native-seed allowlist
(`_POST_BOOTSTRAP_NEW_NATIVE_SEEDS`) deliberately does not auto-trust it on
the HOST engine path, which is why the docker engine (which installs the skill
itself) is the supported path for v6.56.0 bridge runs.

## Addendum (2026-07-05, campaign tick)

4. `run_clbench_bridge_agent.v6560.patch` — pass `run_index` to the task ctor
   only when its signature accepts it (`CodebaseAdaptationTask` lacks the
   parameter at the pinned runner commit; the DB task has it). Both call
   sites (stateless `_make_build` and the stateful path).
5. Runner-python venv also needs `mini-swe-agent` (import `minisweagent`) for
   the codebase_adaptation domain, and `pip` itself when the venv is created
   by uv (the isolated server's local-dev deps sync shells out to
   `python -m pip`).

## Addendum (v6.74.0)

6. `clb_acceptance_claims.v674.patch` — populate advisory
   `acceptance_claims` in ALL THREE task-body writers (old bridge stateless +
   stateful paths, live-bridge path): the deliverable is the last terminal
   `ANSWER` receipt accepted by `task.step()` through the `clbench_remote`
   skill; a prose restatement is not a deliverable and an INCORRECT receipt
   does not satisfy acceptance. `answer_protocol` deliberately stays EMPTY —
   a final-answer text line would re-legitimise the very channel the grader
   ignores. Also appends the owner-approved knowledge-topic nudge to the
   `steer_note` slot (before `_ACTION_NOTE`, which must stay the prompt tail)
   and updates `test_live_parity.py` accordingly. The poller additionally
   waits (bounded, 60s) for the post-task cost checkpoint on
   `completed`/`degraded` outcomes before reading cost fields
   (`task_cost_finalized` is never emitted for failed/cancelled — those read
   immediately). Apply with `patch -p0` from the external checkout root.

## Addendum (v6.74.5, 2026-07-23)

7. `clb_disabled_tools_env.v6745.patch` — make `DISABLED_TOOLS` read from the
   `CLBENCH_SOLVE_DISABLED_TOOLS` env var (forwarded by `run_clb.py`) instead of
   a hardcoded `[]`. Lets a submittable no-swarm run disable `schedule_subagent`
   from the launcher. Apply with `patch -p0` from the external checkout root
   (after `clb_acceptance_claims.v674.patch`; different hunk, no conflict).
8. Launcher change (no patch file — `run_clb.py` is our own code): `_sanitized_child_env`
   now also forwards `OUROBOROS_TASK_REVIEW_MODE` and `OUROBOROS_MAX_SUBAGENT_DEPTH`
   so a "review required" / bounded-depth submittable run is configurable from the
   launcher instead of silently defaulting to auto / depth 2.

## Addendum (v6.74.5, campaign luna 2026-07-23)

8. `clb_env_campaign_overrides.v6745.patch` — `_docker_launcher._overrides()`
   gains an operator-env override block: `OUROBOROS_RUNTIME_MODE`,
   `OUROBOROS_TASK_REVIEW_MODE`, `OUROBOROS_REVIEW_MODELS`,
   `OUROBOROS_EFFORT_REVIEW`, `OUROBOROS_EFFORT_SCOPE_REVIEW`,
   `OUROBOROS_REVIEW_ENFORCEMENT`, `OUROBOROS_SAFETY_MODE`,
   `OUROBOROS_MAX_SUBAGENT_DEPTH`, `OUROBOROS_MAX_WORKERS`,
   `OUROBOROS_CONTEXT_MODE`, `OUROBOROS_SUBAGENTS` — explicit host exports
   win over the hardcoded CC-parity defaults (runtime=advanced, 3× reviewer
   list, uniform effort).
   Launcher side: `run_clb.py` forwards these knobs from the settings template
   (and fixes CLBENCH_SOLVE_DISABLED_TOOLS string-vs-list join). Without the
   patch pair a campaign asking runtime=pro / single low-effort reviewer /
   review=required silently ran on parity defaults.

## Addendum (v6.76.0, 2026-07-25)

### Apply order (unambiguous)

Both patches below were generated against a clone that ALREADY carries every
patch above, so they are applied LAST. In particular, on
`run_clbench_bridge_agent.py` they must come **after these three, in this
order**: `run_clbench_bridge_agent.v6560.patch` →
`clb_acceptance_claims.v674.patch` → `clb_disabled_tools_env.v6745.patch`.
On `_docker_launcher.py` they come after `_docker_launcher.v6560.patch` →
`clb_env_campaign_overrides.v6745.patch`. Apply with `patch -p0 < <file>` from
the external checkout root (a clean clone at `549998d` plus the patches above).

9. `clb_multi_instance_outcomes.v6746.patch` — the stateful bridge loop takes
   the **whole** `instance_outcomes` list from `GET /_outcome` instead of only
   the row whose `instance_index` matches the question the agent was handed.
   The shim already collects outcomes correctly and non-destructively with the
   official runner's own helpers (`_collect_step_outcomes` /
   `_upsert_instance_outcomes` over `task.get_instance_outcomes()`), and ONE
   `task.step()` can close SEVERAL instances; the old single-row read dropped
   every co-resolved instance, because `advance_question()` then walked past
   them and no `q###/task_outcome.json` was ever written for them. Recovered
   rows are labelled `ouroboros_status="auto_resolved_no_agent_turn"` and carry
   NO agent-turn artefacts (no `prompt.txt`, no `ouroboros_task_final.json`, no
   `absorb.json`) and `cost_usd: null` — honest labelling, never a silent
   backfill. An ALREADY recorded row is never rewritten except to fill a still-null
   reward that only materialised at `evaluate()` time (marked
   `reward_source: "late_instance_outcomes_fill"`; status and cost are preserved).
   Consequence, ACCEPTED and disclosed (owner decision): the loop's
   `steps` counter counts agent turns, so the recorded instance count can
   overshoot `--num-instances` by the size of the last co-resolved group. That
   is deliberate — dropping already-scored official rewards to hit the window
   exactly would be worse. Report the scored count from `result_index.jsonl`,
   not the requested window. `run_clb.collect_results` needs no change: it
   reads the same per-question `task_outcome.json` schema.
   BOTH writers of that file go through `_atomic_write_json` (temp file in the
   same directory, flush+fsync, close, then one `os.replace`), which the patch
   adds. `task_outcome.json` is the ONLY record that an instance existed, so a
   truncating `write_text` interrupted mid-flight does not lose an update, it
   loses the ROW: `collect_results` would parse `{}` and the instance would
   disappear from the denominator this patch exists to preserve. The
   late-reward fill rewrites rows that are already VALID, which is when that
   costs the most. A LOCAL seam, not `devtools...manifests.write_json`: this
   module lives in the external checkout and only reaches the Ouroboros clone
   through `_launcher`'s call-time `sys.path` insert, and that helper also
   imports `ouroboros.utils`, which the CL-Bench runner venv does not install.
10. `clb_docker_runtime_attestation.v6746.patch` — runtime attestation for the
   **docker** engine path. The host-engine path is attested through
   `IsolatedServer._wait_ready()`, but `_docker_launcher.DockerOuroborosEngine`
   is only a "thin stand-in for IsolatedServer" (own `_DockerServer`, own health
   gate) and never calls `_wait_ready`, so the supported docker path would
   otherwise run unattested. The patch adds `_attest_runtime()`, called right
   after the health+settle gate and BEFORE any paid task, which calls the shared
   helper `devtools.benchmarks.common.manifests.runtime_attestation(base_url,
   repo_dir, *, expected_version="", timeout=10)` (both the HTTP
   `runtime_version` and the local repo head). `repo_dir` is a REQUIRED
   POSITIONAL — it is how the commit half of the attestation is reported — and
   the patch passes the mounted clone, whose `VERSION` is also handed over as
   `expected_version`. The HELPER owns the policy: skew, an unreachable runtime,
   or an unreadable commit aborts the run, and the named escape
   `OBO_ALLOW_EVOLVED_VOLUME` (`1`/`true`/`yes`) downgrades it to a record with
   `overridden: true` for a deliberately evolved `/obo/repo` volume. The patch
   keeps no second copy of that decision; it stores the record on the engine as
   `runtime_attestation` and prints it.
   **Requires an Ouroboros bench clone at v6.75.0 or newer** — that is where
   `runtime_attestation` lands. On an older clone the patch fails closed with an
   explicit ImportError message naming the requirement, so do not apply it to a
   pre-v6.75.0 bench clone.

## Addendum (v6.81.0, 2026-07-26)

11. `clb_multi_instance_outcomes.v6746.patch` was **regenerated in place** (no new
    file, same apply order) to publish the RUNTIME's own terminal reason in each
    `q###/task_outcome.json`. `_write_instance_outcome` gains a keyword-only
    `runtime_result` and a `runtime_outcome` column projected by a new local
    `_runtime_terminal_disclosure` helper; the agent-turn call site passes `final`,
    the co-resolved call site passes nothing (it had no turn, so the row honestly
    reads `{"available": false}`). WHY: `ouroboros_status` alone cannot carry it —
    a question the per-task USD reservation rail truncated reports status
    `failed` exactly like a genuinely wrong answer, so an aggregator could not tell
    a cost-truncated question from a capability failure. Reward, success and
    `cost_usd` are untouched; this ADDS disclosure.
    The helper is a deliberate LOCAL mirror of
    `devtools.benchmarks.common.result_index.runtime_terminal_disclosure` (the SSOT
    for the vocabulary) for the same reason `_atomic_write_json` is local: this
    module lives in the external checkout and only reaches the Ouroboros clone
    through `_launcher`'s call-time `sys.path` insert. The mirror is ENFORCED, not
    merely requested: `tests/test_benchmark_provenance_v6810.py::
    test_clb_operator_patch_mirrors_the_truncation_ssot_verbatim` parses the
    truncation vocabulary out of THIS patch file and fails if it diverges from
    `RUNTIME_TRUNCATION_REASON_CODES`. A "keep in sync" comment alone did not
    hold — the first cut of both copies listed four codes the runtime never emits
    and missed `round_limit`/`deadline_local`.
    Launcher side (`run_clb.py`, our own code, no patch file):
    `collect_results` carries `runtime_outcome` into `results.json` /
    `result_index.jsonl`, defaulting to `{"available": false}` for rows written
    before this patch — a stated gap, never a silent absence.

## Addendum (v6.81.0 submission campaign, 2026-07-28)

9. `adapter_official_submission.v681.patch` — the ENTIRE merged submission adapter as one
   delta against upstream `5f8c50eb` (post cohort-fix). Base: colleague's official-path
   adapter (final_answer_delivery, live recovery, 47 parity tests) + all v6.81.0 campaign
   ports (env override loop incl. runtime/safety/context/split efforts, vision pin,
   CLBENCH_SOLVE_DISABLED_TOOLS, per-question cap, docker runtime attestation, credential
   scoping) + NEW: engine-task cost -> CL-Bench UsageEvent (official manifests otherwise
   under-report cost ~1000x; colleague's manifest showed $0.54 vs ~$530 actual).
   Apply with `git apply` onto a clean checkout of upstream 5f8c50eb. This patch
   SUPERSEDES patches 1-8 for the official-path (run-all) flow; the bridge-path patches
   remain for archaeology. Working tree: /mnt/data/a.razzhigaev/clb_official_v681/bench,
   branch `ouroboros-submission`.

   Current Ouroboros launchers also pass the canonical one-actor
   `OUROBOROS_SUBAGENTS` bytes through this adapter. The patch no longer authors
   legacy Heavy; it transports the value produced by the Phase-1A encoder.

## Addendum (luna ablation campaign, 2026-07-31)

10. `adapter_luna_ablation.v6870.patch` — delta on top of `adapter_official_submission.v681`
    (i.e. against submission commit `a691cf3`) for the harness ablation that runs Ouroboros
    and the Codex CLI on the SAME model, `openai/gpt-5.6-luna`. Two changes:
    - `src/systems/ouroboros/_live_bridge.py`: `_format_repair_action` referenced `_rb`
      without the lazy import every other use site in that module performs, so the
      last-resort format-repair branch raised `NameError` the moment it was reached —
      the exception escaped `respond()`, `_abandon_task()` never ran, and the question
      was scored as an empty action. One occurrence in 3204 deliveries on sonnet-4.6;
      the branch matters far more on a weaker model, which is what this campaign runs.
    - `src/systems/codex/system.py`: an optional custom provider (`provider_base_url`,
      `provider_env_key`, `provider_id`) so the Codex arm can be pointed at the same
      OpenRouter endpoint as the Ouroboros arm instead of native OpenAI — otherwise the
      comparison confounds harness with provider. Codex refuses to redirect its built-in
      `openai` provider id, so this declares a NEW `model_providers.<id>`, and this CLI
      generation only loads a custom provider with `wire_api="responses"`. On that path
      `auth.json` is not written (the key comes from `env_key`) and the recorded usage
      provider names the route actually taken. Default behaviour is unchanged.
    The provider is declared once in `CODEX_HOME/config.toml` at container start, NOT as
    per-turn `-c` flags: those share argv with the prompt, and on the longest questions
    of `database_exploration` the pair overflowed the OS limit — `docker exec` died with
    `OSError: [Errno 7] Argument list too long`, which is not a recoverable system error,
    so it killed the whole task after its five rollouts had already scored. Declaring it
    in config.toml keeps per-turn argv smaller than upstream's.
    Verified: bench suite 649 passed / 7 skipped (`tests/test_bsm_online_model.py` needs
    task-specific data and fails to collect in any checkout, before and after).
    Public implementation for the ablation run: branch `ouroboros-luna-ablation` of
    `razzant/continual-learning-bench`. PR #10's `ouroboros-submission` branch is left
    untouched so the sonnet-4.6 submission keeps matching the code that produced it.

## Addendum (6.113.5 evolution campaign, 2026-09-06)

12. `adapter_evolution_campaign.v61135.patch`: the adapter as it stood after the
    evolution-arm pilots of 2026-08-27 to 2026-09-06 on Ouroboros 6.113.5, as one delta
    against PR #10 head `a691cf3` (branch `ouroboros-submission` of
    `razzant/continual-learning-bench`). Apply with `git apply` onto a clean checkout of
    `a691cf3`. It already contains the one-line `_rb` import fix of
    `adapter_luna_ablation.v6870.patch` and does not touch `src/systems/codex/system.py`,
    so do not also apply the luna patch's `_live_bridge.py` hunk.

    This delta diverges from the current code and is published for its methodology, not
    as a drop-in adapter. It branched before the regeneration of
    `adapter_official_submission.v681.patch` that added the `OUROBOROS_SUBAGENTS` transport
    from `run_clb.py` and removed the legacy HEAVY/CODE pins: it does not apply on top of
    that patch (conflict in `_docker_launcher.py`), keeps the old pins, and writes its own
    one-row roster instead of forwarding the launcher's. Its restart-refusal handling
    targets an admission fence that exists only in the local engine build it ran on. It
    was never run on a 7.x engine, and the 7.4.5 code shows two breaks: the injected
    `remote_work` manifest lacks `plugin_api: "2.0"`, so the engine refuses its native-seed
    trust and the live loop has no tools; and the task record no longer carries `cost_usd`
    or `cost_usd_with_children`, so engine-task UsageEvents lose their cost. On 6.113.5 as
    on 7.4.5 a fresh image build is expected to fail, because the engine's
    `requirements.txt` has been a pointer to a lock file the Dockerfile does not copy since
    v6.97.0; the 2026-09-03 run used a previously built image.

    The methodological changes, which are what this delta is for:
    - The next question waits for the evolution round. The boundary wait used to end while
      a cycle was still running (over the 12 completed cycles of the 2026-09-03
      treatment run, 464 s at the median and 920 s at most; the pilots capped the wait at
      300 s), and 14 of 22 questions in the 2026-09-03 treatment
      run were submitted into a running cycle, where the restart that cycle requests can
      kill them. The engine starts an evolution task only when its queue is idle, so the
      overlap came entirely from the adapter. `on_instance_boundary` now waits until a
      host-visible predicate says the engine is quiet: no promotion request, evolution
      disarmed, no queued or running task, no active direct-chat activity (the post-restart
      auto-resume task runs as one), no pending or claimed restart marker, and no
      unresolved restart obligation. It needs two consecutive quiet samples, has one 1800 s
      budget per boundary, and on expiry logs the blockers and submits anyway.
    - On the forced arm, evolution cycles come only from the benchmark boundary: the adapter
      sets `OUROBOROS_POST_TASK_EVOLUTION_CADENCE=off` there. Under the adapter's default
      `every_n:1` (the engine's own default is `llm`) every finished task, including the engine's own post-restart auto-resume task, files
      the next promotion, so cycles chained between questions (13 evolution tasks against
      11 boundaries in the 2026-09-03 treatment run).
    - A restart no longer silently costs a question: the drain deadline is raised to
      1800 s (engine default 120 s), a question cancelled by a restart is retried once, and
      submits wait out a pending restart. The earlier form of this gate held 8 submits in
      the 2026-09-03 treatment run.
    - The answer is located with the same code as the official Claude Code and Codex
      adapters (`_json_candidates` and `_parse_action_text`, copied verbatim and pinned by a
      test) over the task's assistant transcript, so a complete answer followed by a
      delivery-control envelope is not lost.
    - The sales sandbox outlives the 2 h library default (`container_timeout` raised to
      24 h at import), which every stateful Ouroboros sales rollout of the submitted run
      exceeded; pgasawa/continual-learning-bench#22 proposes the bench-side fix.
    - Reviewer slots are pinned to the solve model (`OUROBOROS_REVIEWER_SLOTS`: the triad as
      `api_chat` rows, scope and advisory through a same-model scout row). Without the pin
      the isolated settings inherit the host's structured slots, which override the flat
      single-model review keys.
    - Turn-1 responses attest the resolved configuration in `Response.metadata`.

    Last end-to-end run: 2026-09-03, CL-Bench `run-all` groups
    `2026-09-03T05-22-33.505496Z` (control) and `2026-09-03T05-22-33.505495Z` (treatment),
    `sales_prediction`, `openrouter/anthropic/claude-sonnet-4.6`,
    on Ouroboros 6.113.5 built from a local engine-fix branch (head `1ffa14d5`, not
    published; public ancestor `8d13373b`), runtime attestation
    `clone_version=6.113.5 head=5fd3157e`. Active in that run: the earlier restart gate and
    retry, the 1800 s drain, the reviewer-slot pin, the sandbox lifetime override and the
    config attestation. Added afterwards and exercised by unit tests only: the
    official-parser answer locator, the boundary readiness wait, the cadence `off` and the
    current restart gate. The patch also rewrites the post-submission section of the
    adapter's `METHODOLOGY.md` with the full account.
    Verified: the patch reproduces its source tree byte for byte on `a691cf3`;
    `tests/test_ouroboros_submission_ports.py` 140 passed and
    `tests/test_ouroboros_live_parity.py` 76 passed in the bench venv.
