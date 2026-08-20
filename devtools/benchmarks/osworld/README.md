# OSWorld Devtools

This directory contains OSWorld utilities for both logs-only audits and a
submission-shaped external step-loop runner. Official OSWorld reproducibility
still requires a runnable OSWorld checkout plus VM/desktop control
infrastructure; public verified leaderboard claims require the official
verification path (the OSWorld 2.0 maintainers run agent code on their side).

Aligned upstream (pinned; see `ALIGNED_UPSTREAM` in `run_step_agent.py` and
`METHODOLOGY.md` §0): OSWorld 2.0,
<https://github.com/xlang-ai/OSWorld-V2> @
`c261cb57a699bd18db128787ca4e71b749141762` (2026-06-30), paper
arXiv:2606.29537. Protocol highlights verified from that commit: 500-step
budget (`scripts/bash/run_multienv_claude.sh`; the bare `run.py` default of 15
is legacy), `pyautogui` action strings with `WAIT`/`DONE`/`FAIL` specials,
VM-state-only evaluators, and the `show_result.py` result layout
`<result_dir>/<action_space>/<observation_type>/<model>/<domain>/<example_id>/result.txt`.

Files:

- `normalize_logs.py` indexes logs-only bundles for analysis.
- `schemas.py` validates the known logs-only JSON layout.
- `settings_base.json` — bench settings template (single-model, secrets
  blank, `OUROBOROS_MAX_WORKERS=4`, `OUROBOROS_SAFETY_MODE=light`,
  `OUROBOROS_RUNTIME_MODE=pro`; see the benchmarks index "Bench-Template
  Scaffold Defaults").
- `METHODOLOGY.md` — official 2.0 protocol, scaffold disclosures, honest
  runnable-vs-skeleton status, the cu_bridge final_answer lesson, and (§7.9) the
  pre-registered multi-lane/dedup rules.
- **Multiple lanes are supported; the lane-script GENERATOR is not in this
  release.** Overlapping runs are a supported configuration and the smoke uses
  them: several operator-written lane scripts, each invoking
  `run_cu_bridge_agent.py --claim-dir <shared>` against its own isolated bench
  server, over a shared results tree. What was built during this phase and
  EXTRACTED before release is the convenience generator for those scripts
  (`gen_lanes.py`, lane port binding, `lanes.json`); nothing here generates lane
  scripts, allocates lane ports or starts more than one bench server. The
  `--claim-dir` mechanism is what makes overlapping attempts safe and is not
  lane-specific — append-only resumes and retry passes over a shared results tree
  need the same ownership answer. See METHODOLOGY.md §7.9.
- `operator_patches/` — unified diffs for the THIRD-PARTY OSWorld checkout (never
  a fork, never a commit into it; never tasks/evaluators/scoring). Currently the
  docker provider's `LOCK_TIMEOUT` 10s→60s, which together with our constructor
  retry keeps concurrent lanes from dying on `/tmp/docker_port_allocation.lck`.
- `osworld_adapter_skeleton.py` refuses to run unless the official environment,
  live Ouroboros server, computer-use payload, and output-root isolation are all
  present. It also requires `unix_computer_use` to have a fresh executable review
  under the blocking review gate (`pass`/`advisory_pass` legacy aliases or
  canonical `clean`/`warnings`) and then pass `skill_readiness_for_execution()`
  for enabled state, grants, and dependencies. It writes fail-closed
  ledger/manifest artifacts for blocked preflights when the output root is
  outside `repo/` and runtime `data/`.
  The readiness probe uses the same runtime skill loader/readiness gate and may
  initialize empty state directories under the declared isolated data root. If
  `--data-root` is omitted, the CLI uses `<output-root>/isolated_data`; it must
  not point at live `/Users/anton/Ouroboros/data` for smoke runs.
- `run_step_agent.py` is the external OSWorld step-loop runner. It resets an
  official OSWorld VM, saves each screenshot beside the task trajectory under
  the result directory, calls `ouroboros run --attach <screenshot>` for the next
  structured action, executes those actions through `env.step(...)`, and records
  the official trajectory plus denominator-preserving ledgers. It is the runnable
  adapter; the skeleton remains a stricter installed-agent preflight path. Its
  owner leaves are re-exported by it, so behaviour, flags and reward semantics
  are unchanged: `step_agent_common.py` (run configuration dataclasses and two
  shared primitives), `step_agent_env.py` (aligned upstream pin, provider
  preflight, DesktopEnv construction/teardown, live-server guard),
  `step_agent_claims.py` (cross-lane task claims and the scored-claim ledger of
  METHODOLOGY §7.9), `step_agent_actions.py` (action translation and the
  `WAIT`/`DONE`/`FAIL` specials) and `step_agent_policy.py`
  (`OuroborosStepAgent`). `_preflight` deliberately stays in the launcher: its
  runtime attestation is pinned to that file by the devtools test suite.
- `run_cu_bridge_agent.py` is the **persistent-agent** OSWorld runner: it resets
  an official VM, publishes the VM HTTP target into the bench data dir's
  `unix_computer_use` skill state, submits ONE Ouroboros task (`--memory-mode
  empty`), and lets the agent drive the VM through the skill's `osworld_http`
  backend until it finishes. `reset()`/`evaluate()` are the official ones. This
  is the Terminal-Bench / Pointer shape — see the cu_bridge details below and
  METHODOLOGY.md §7 for the protocol deltas that make it NOT the official
  step-loop. Its owner leaves carry the parts that are not the launcher itself
  and are re-exported by it, so behaviour, flags and reward semantics are
  unchanged: `cu_bridge_runtime.py` (shared bench-server call and terminal-answer
  reading), `cu_bridge_prompts.py` (gate/working preambles and acceptance
  claims), `cu_bridge_tool_policy.py` (core-tool allowlist, computed host
  denylist, GUI action set, denied connection tools), `cu_bridge_gate.py` (the
  read-only premise gate) and `cu_bridge_budget.py` (step/round budgets, proxy
  configuration, dataset and step-claim refusals, disclosure counters).

Important step-loop details:

- `--max_steps` defaults to 500 (the OSWorld 2.0 protocol budget). Inline
  checkpoint evaluations (official 150/300-step curves), multi-phase tasks,
  the user simulator, and `recording.mp4` are NOT implemented — final-state
  evaluation only.
- Per-example artifacts follow the official layout consumed by upstream
  `show_result.py`: `traj.jsonl` rows with `step_num`/`action_timestamp`/
  `action`/`response`/`reward`/`done`/`info`/`screenshot_file` (adapter
  extras are namespaced under `adapter_debug`), post-action
  `step_<n>_<ts>.png` screenshots, `result.txt`, and `result.json` when
  `env.evaluate()` returns a dict.
- Preflight validates the external environment and fails loudly with what is
  missing: OSWorld checkout variant (`v1`/`v2` markers) and git commit vs the
  pinned upstream, `desktop_env` presence, provider availability (`vmware`:
  vmrun + `.vmx`; `docker`: reachable daemon; cloud providers are
  unsupported), Ouroboros server/settings/model key, and output-root
  isolation.
- The evaluator checks VM STATE, not chat: the prompt says so explicitly, and
  the runner captures the agent's terminal message into `final_answer` /
  `terminal_action` / `infeasible_declared` in `task_outcome.json` and
  `result_index.jsonl` so "answered in chat" is auditable instead of an empty
  field (cu_bridge sample-60 lesson; METHODOLOGY.md §4).
- Screenshots are passed as native image attachments to the model. `vlm_query`
  remains a fallback for non-vision models.
- Shell actions are written into a temporary in-VM script and executed by path;
  the raw command is base64-encoded inside the action snippet. This prevents
  `pkill -f <pattern>` from matching the wrapper process's own argv.
- The prompt is in-app first: when a task names an application, work in that
  application or reopen/verify direct file edits in that application before
  `done`.
- The agent may return a `notes` field; the runner carries recent notes across
  otherwise stateless Ouroboros steps.
- `claude_code_edit` is withheld per step by default (`--disable-tools`).

Important cu_bridge details (`run_cu_bridge_agent.py`):

- **Start the isolated Ouroboros server with `unix_computer_use` ALREADY
  ENABLED.** The skill declares `net`, so it is NOT in the launcher's native
  auto-enable class; the server loads enabled+reviewed extensions only at
  startup (`reload_all`). The runner's `_enable_skill` writes `enabled.json`,
  which a server started BEFORE that write will not hot-reload — the skill's
  `ext_*` tools then come back "Not found" and the agent declares the task
  infeasible. Seed the skill state (enabled) before `server.py` starts (the
  parallel orchestrator does this by starting a fresh isolated server per task
  after seeding), or restart the server after enabling.
- One persistent Ouroboros run per task drives the VM through
  `unix_computer_use` (osworld_http backend), instead of the host driving
  `env.step`. GUI actions therefore go straight to the guest `/execute` server
  and are NOT recorded in the official `DesktopEnv.action_history` /
  `traj.jsonl` — only the translated `FAIL` is (METHODOLOGY.md §7).
- Observation modality is **screenshot-only by default**: `ax_tree` (the
  on-demand accessibility-tree fetch) is added to the per-task disabled tools
  unless `--allow-a11y` is passed. A run with `--allow-a11y` must be disclosed
  as "Additional a11y tree used: Yes".
- `remote_exec` (guest shell) stays available — disclose "Additional
  coding-based action: Yes" (matches the Pointer leaderboard precedent). The
  GUI-first preamble is advisory, not enforced. Note `remote_exec` is a
  computer-use SKILL tool acting on the VM guest, not a host tool.
- Host-tool lockdown: the OSWorld instruction is untrusted, so the task is
  submitted with a computed denylist = all core tools minus a small allowlist
  (skill discovery/enable, `view_image`, read-only inspection). Every host
  execution/mutation/VCS/GitHub/service/self-mod/chat surface is blocked by
  construction (`_ALLOWED_CORE_TOOLS` / `_host_denied_tools`); the VM is driven
  only through the `unix_computer_use` skill's ext_* tools. The skill's
  connection-SWITCHING ext tools (`add_connection`/`activate_connection`/
  `use_local`/`clear_active_connection`) are ALSO denied so the task cannot
  switch the pinned VM connection to `local` and drive the host desktop
  (`_DENIED_SKILL_EXT_TOOLS`); read-only `list_connections`/`test_connection`
  and the VM-control tools stay.
- Guards: refuses `--ouroboros-url` on the live desktop port 8765 unless
  `--allow-live-server`; unconditionally refuses a `--data-dir` inside the live
  `~/Ouroboros/data` root (publishing a bench connection there would hijack the
  owner's real skill).
- `task_outcome.json` records disclosure counters (`budget_counters`:
  `llm_rounds`, `screenshots`, `gui_action_calls`, `remote_exec_calls`) and
  `max_rounds_effective`; the manifest records the OSWorld checkout variant/pin
  (`osworld_checkout`) and `a11y_enabled`. The budget is the Ouroboros server's
  `OUROBOROS_MAX_ROUNDS` (default 200) plus `--task_timeout_sec`; this is NOT a
  100-step leaderboard cap — report both.
- The VM sudo password is injected into `prompt.txt` (official OSWorld practice;
  `mm_agents/prompts.py`). Keep run artifacts access-controlled.
- Proxy: for `"proxy": true` tasks the runner enables OSWorld's proxy pool only
  when a proxy config file exists; otherwise the task runs without proxy —
  disclose this in the campaign report.

Example cu_bridge smoke:

```bash
# ISOLATED bench server (fresh OUROBOROS_DATA_DIR, non-default port). --data-dir
# must NOT be the live ~/Ouroboros/data. Screenshot-only unless --allow-a11y.
python devtools/benchmarks/osworld/run_cu_bridge_agent.py \
  --osworld-root /path/to/OSWorld \
  --provider_name docker \
  --path_to_vm /path/to/Ubuntu.qcow2 \
  --task evaluation_examples/examples/multi_apps/48d05431-6cd5-4e76-82eb-12b60d823f7d.json \
  --result_dir results/osworld_cu_bridge \
  --data-dir /path/to/bench_data \
  --target-file /path/to/bench_data/cu_target.txt \
  --ouroboros-url http://127.0.0.1:8780
```

Example step-loop smoke:

```bash
# Point at an ISOLATED Ouroboros server (fresh OUROBOROS_DATA_DIR, non-default
# port) — the runner refuses the live desktop URL http://127.0.0.1:8765 unless
# you explicitly pass --allow-live-server for a local debug run.
python devtools/benchmarks/osworld/run_step_agent.py \
  --osworld-root /path/to/OSWorld \
  --task evaluation_examples/examples/multi_apps/48d05431-6cd5-4e76-82eb-12b60d823f7d.json \
  --result_dir results/osworld_step_agent \
  --ouroboros-url http://127.0.0.1:8770 \
  --model anthropic/claude-opus-4-7 \
  --max_steps 5
```

For current official OSWorld comparisons, run on the official
environment/architecture. Google Drive tasks need `client_secrets.json`; if it
is unavailable, use the documented 361-task exclusion path rather than counting
harness setup crashes as model failures.

## Feasibility gate (`--feasibility-gate`, opt-in, off by default)

A read-only premise phase that runs BEFORE the working task. It is a separate Ouroboros
task whose mutating GUI tools (`click`, `move`, `left_click_drag`, `mouse_down`,
`mouse_up`, `type_text`, `key`, `hold_key`, `scroll`) are **absent from the capability
envelope**, not merely discouraged — so the premise cannot be manufactured while it is
being judged. Reading stays available (`screenshot`/`view_image`/`window_list`/`wait`)
and `remote_exec` stays for read-only probes; read-only there is an instruction, not an
enforcement, because deciding in code whether a shell command reads or writes would be
exactly the kind of pattern gate that BIBLE P5 forbids for a semantic decision.

The phase works through a STRUCTURED RUBRIC — requested action → pre-existing referent
→ does absence block the action → is the missing thing acquirable by means the
instruction does not forbid → does a "set X to Y" target merely store the value →
unbound placeholders — and answers one of three words on its last line: `INFEASIBLE`,
`PROCEED`, `UNDETERMINED`. The rubric replaced a description-by-example (v6.81.1): the
v6.81.0 false kills all judged whether the OUTCOME would be meaningful ("no saved Etsy
password to check", "the font is not installed") instead of whether the requested
ACTION was performable.

**It fails OPEN, on a SINGLE verdict.** Only an explicit `INFEASIBLE` ends the example
(translated into the official `env.step("FAIL")` through the same path a self-declared
infeasibility takes, so the claim-marker and scoring sequence is not duplicated).
`PROCEED`, `UNDETERMINED`, an unparseable answer, a timeout whose cancel confirmed, a
crashed phase or any exception all fall through to the full-capability phase, which
keeps its own mid-task `TASK_INFEASIBLE` path. The v6.81.0 revision ran a confirming
CHALLENGER round on every `INFEASIBLE`; its own full-run ledger removed it — 20
invocations, 0 feasible tasks saved, 1 officially-infeasible task lost, 215 worker
rounds burned, and it confirmed all four of the gate's false kills. An identical-prompt
re-read produces correlated errors, not an independent check.

The one condition that does NOT fail open: a premise round whose cancel did not confirm.
That leaves a zombie session sharing the lane's server and skill connection file, able
to act on the VM the worker is scored on (and on the lane's next task). The attempt
aborts as a typed infra row (`blocked` / `gate_cancel_unconfirmed`, exit 2 so the lane —
and with it the zombie's server — dies; the claim is released for a clean retry).

Known limit, stated plainly: this closes the GUI vector, not the shell. `remote_exec` is
a general shell and is read-only in this phase by instruction only — enforcing it in code
would be the pattern gate the constitution forbids for a semantic decision. Two
mitigations follow. (1) `feasibility_gate.json` carries the round's FULL tool trace
(verbatim args, no previews) for offline audit of that promise. (2) The working phase is
re-reset after a PROCEED/UNDETERMINED verdict, so anything the premise phase touched is
discarded before the state that gets scored.

**Every reset republishes the VM endpoint — this is the repair that mattered.**
`DockerProvider.revert_to_snapshot` stops the container and `start_emulator` reallocates
ports, so the VM's address changes on every reset. The 2026-07-28 v1 smoke published it
once, before the gate (83/83 task dirs had `bridge.json` older than their gate record),
so the working phase kept driving the pre-gate address — which, with 16 lanes allocating
from one port range, another lane's container could already own. Feasible-control mean
fell 0.737 → 0.459, with workers reporting empty Desktops, missing task files, and in one
case acting on a different task's presentation entirely. Republishing removed the class:
0 regressions in v2 against 9 at the comparable stage of v1.

**Both resets are also VERIFIED** (`_reset_verified`) against a second, independent
fail-open — this one inside OSWorld: when the guest server misses the setup probe window
`reset()` skips every setup step, logs "Environment setup complete." and returns a
pristine VM with no exception. The runner asserts `is_environment_used` whenever the task
config is non-empty, keeps the screenshot probe (same HTTP path the agent's tools use),
forces the snapshot revert before every retry, and turns exhaustion into a typed infra
row (`reset_unverified`, reward `null`, claim released) instead of a capability zero.
Stated honestly: this guard has not fired in production yet (v2: 24 post-gate resets, 0
retries) — it is defence in depth, not the measured fix, though it does close a real
pre-existing flaw that affects ungated runs on their single reset.
`reset_verification.json` records attempts and the captured `desktopenv` log tail.

Cost and disclosure: a gated run posts UP TO two tasks per example — one when the gate
ends the example, two when the working phase runs — so the manifest reports
`one_run_per_task: false`, `feasibility_gate_phase: true` and
`feasibility_gate_challenger: false` (v6.81.0 manifests say `true` — the flag exists so
readers of both runs see the scaffold difference), and the per-example verdict lands in
`feasibility_gate.json`. Expect roughly double the per-example warm-up and
acceptance-review cost on examples whose working phase runs. See METHODOLOGY.md §7 (4c).

**Validate before trusting it.** The failure mode that matters is a false `INFEASIBLE` on
a feasible task: that scores a hard zero. Measure BOTH the false-INFEASIBLE rate and the
paired score delta against a prior run on a stratified control set of feasible tasks —
not just the recall on infeasible ones — before enabling it for a scored run, and
interleave the classes in the task list: a blocked layout delivers the flattering class
first and the deciding class last.
