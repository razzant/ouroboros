# OSWorld Methodology Notes

These notes state the official OSWorld 2.0 contract first, then the Ouroboros
scaffold disclosures, then the honest status of what is runnable here today.
Official OSWorld harnesses and `env.evaluate()` outputs remain the scoring
authority; the sidecar ledgers in this directory are audit artifacts.

## 0. Pinned upstream (authoritative alignment target)

| Field | Value |
| --- | --- |
| Repo | <https://github.com/xlang-ai/OSWorld-V2> |
| Commit | `c261cb57a699bd18db128787ca4e71b749141762` (2026-06-30) |
| Paper | arXiv:2606.29537 — "OSWorld 2.0: Benchmarking Computer Use Agents on Long-Horizon Real-World Tasks" |
| Legacy repo | <https://github.com/xlang-ai/OSWorld> (OSWorld 1.0 / OSWorld-Verified) |

`run_step_agent.py` records this pin as `ALIGNED_UPSTREAM` in every run
manifest, and its preflight reports the local checkout's actual git commit and
variant (`v1`/`v2`, detected by `evaluation_examples/test_v2.json` vs
`test_all.json`) so a drifted checkout is visible instead of silent.

## 1. What the bench measures

OSWorld executes an agent against a real Ubuntu desktop VM. Each task ships a
per-example JSON config with a natural-language `instruction`, a `config`
setup block, and an `evaluator` block. After the episode the harness calls
`DesktopEnv.evaluate()`, which scores **VM state only** — files at exact
paths, in-application/document state, the browser's active tab URL, OS
configuration — via result/expected getter functions and metric functions
(`desktop_env/desktop_env.py::_evaluate_with_evaluator` at the pinned commit).
The agent's chat text is never read by any evaluator. The single
"message-like" channel is the special `FAIL` action: tasks with
`evaluator.func == "infeasible"` score 1.0 iff the last action is `FAIL`, and
a `FAIL` on any feasible task scores 0.

- OSWorld 1.0: 369 tasks (361 without the Google Drive subset when
  `client_secrets.json` is unavailable), binary rewards, historical step
  budgets of 15 (run.py default) / 50 (common community setting).
- OSWorld 2.0: 108 long-horizon tasks (`evaluation_examples/test_v2.json`),
  fine-grained partial rewards averaging ~27.25 checkpoints per task scored
  against the final environment state (order-free); model-based judging
  contributes 11.53% of total score and no task relies on it for more than
  50% (paper §evaluation).

## 2. Official 2.0 protocol (from the pinned sources)

- **Step budget 500.** The official launch scripts pass `--max_steps 500`
  (`scripts/bash/run_multienv_claude.sh`); the paper reports 150/300/500-step
  curves via inline checkpoint evaluations
  (`--checkpoint_eval_mode inline --checkpoint_steps 150,300`,
  `lib_run_single.py::_run_inline_checkpoint_eval`). The bare `run.py`
  argparse default of 15 is legacy. One step = one `agent.predict()`
  observe→act exchange; a step may emit several actions.
- **Action space.** `pyautogui` code strings (plus `claude_computer_use` for
  Anthropic-native runs) executed through `env.step(action, pause)`; special
  strings `WAIT` / `DONE` / `FAIL` terminate or pause.
- **Observations.** Screenshot bytes per step (paper headline runs are
  screenshot-only); the env can also return an accessibility tree
  (`require_a11y_tree`).
- **Result layout** consumed by official `show_result.py`:
  `<result_dir>/<action_space>/<observation_type>/<model>/<domain>/<example_id>/`
  containing `traj.jsonl` (rows: `step_num`, `action_timestamp`, `action`,
  `response`, `reward`, `done`, `info`, `screenshot_file`),
  `step_<n>_<ts>.png` post-action screenshots, `result.txt` (final float
  score — the scoring surface), `result.json` (when `evaluate()` returns a
  dict), `checkpoint_results.json` (inline checkpoint mode), `runtime.log`,
  `recording.mp4`.
- **Providers.** Docker (Linux hosts with KVM) and AWS are the officially
  imaged providers for 2.0; VMware/VirtualBox paths exist in-code. VM login
  `user` / `osworld-public-evaluation`.
- **Leaderboard submission.** There is no artifact-upload pipeline: the
  maintainers schedule a meeting, run the agent code on their side, and
  require the agent implementation under the OSWorld framework plus a
  disclosure report (OSWorld-V2 README). Locally produced numbers are
  therefore *self-reported* by definition; keep the official result layout so
  they can be reproduced/verified.

## 3. Ouroboros scaffold disclosures (v6.55.0 sprint decisions)

- `OUROBOROS_MAX_WORKERS=4` — same-model decomposition slots within one task,
  never independent attempts with selection (pass@1 claims hold).
- `OUROBOROS_SAFETY_MODE=light` — the OSWorld VM is a disposable jail;
  deterministic guards stay on.
- `OUROBOROS_RUNTIME_MODE=pro` — the agent acts on an isolated VM, not the
  live system body.
- `claude_code_edit` disabled per step (`--disable-tools claude_code_edit`) —
  benches measure the single-model Ouroboros harness.
- Single-model: solver/review slots all point at the same model, local routing is
  explicitly off, and `settings_base.json` carries exactly one canonical
  Available-subagent API row plus an authoritative API-only triad/scope panel on
  it. The Claude-SDK advisory is explicitly disabled rather than populated with
  a provider-routed model id. No default scout, second provider family, or session substrate can enter
  the run. Because the step CLI submits to an already-running server, preflight
  verifies that server exposes the same normalized one-row value; client env is
  not accepted as proof. The top-level manifest records the actor returned by the
  target server even for an explicit `--allow-scaffold-mismatch` ablation, rather
  than repeating the local declaration. The CU-bridge runner applies the same
  target `/api/settings` comparison and refuses drift before claiming/booting the
  VM. `settings_base.json` keeps secrets blank; fill keys at run time, never commit.
- Step loop is memory-stateless per Ouroboros call (`--memory-mode empty`);
  cross-step continuity is only the action history + agent `notes` carried by
  the runner prompt.
- The shell action executes via a temp in-VM script (base64-encoded command)
  and deliberately does NOT fabricate `~/.bash_history` entries to satisfy
  terminal-task evaluators (hidden-verifier-knowledge; enforced by
  `tests/test_devtools_benchmarks.py::test_osworld_shell_action_does_not_fabricate_bash_history`).

## 4. The final_answer / VM-state lesson (cu_bridge sample-60 forensics)

In the 2026-06-27 `osworld_cu_bridge_sample60_sonnet46` run, agents that
finished "chat-style" left the Ouroboros `loop_outcome.final_answer` empty
(`final_text` carried the message), so the run's own objective ledger degraded
to `not_evaluated`, and several tasks lost reward because the agent answered
in chat instead of leaving the VM in the evaluator-expected state (e.g.
`is_expected_active_tab` checks the active URL, `compare_table` checks a saved
`.xlsx`). Two structural fixes live in `run_step_agent.py`:

1. **Terminal-message capture.** The per-step JSON schema has a
   `final_answer` field; when the agent emits `done`/`fail` the runner
   persists `final_answer` (falling back to the terminal `response` text) into
   `task_outcome.json` and the `result_index.jsonl` details, together with
   `terminal_action` (`DONE`/`FAIL`/`max_steps_exhausted`) and
   `infeasible_declared`. The audit trail therefore never shows an empty
   answer for an agent that actually answered.
2. **Evaluator-semantics instruction.** The prompt tells the agent explicitly
   that the grader inspects VM state only, that question tasks require
   navigating/leaving the environment in the answering state, and that files
   must be saved to exact paths before `done`. "Answer in chat" is documented
   as scoring zero.

## 5. Honest status: runnable today vs not

Runnable (local code scope):

- `run_step_agent.py` — single-task step-loop against a local OSWorld
  checkout: official actions through `env.step(...)`, official example-dir
  artifact layout (`show_result.py`-compatible), official `env.evaluate()`
  scoring, denominator-preserving ledgers, preflight with checkout
  variant/commit + provider checks. Supports `vmware` and `docker` providers
  only.
- `normalize_logs.py` / `schemas.py` — logs-only bundle audits.
- `osworld_adapter_skeleton.py` — stricter installed-agent preflight path
  (fail-closed; no scoring).

NOT implemented (do not compare as if it were the full official 2.0 harness):

- inline checkpoint evaluations (150/300-step curves) — final-only
  evaluation here;
- multi-phase tasks and the human-in-the-loop user simulator (`ASK_USER`
  rows; our runner maps "no actions" to `WAIT`);
- `recording.mp4` screen recording;
- cloud providers (aws/azure/gcp/aliyun/volcengine) and the official
  parallel `run_multienv*` drivers;
- model-based evaluation env plumbing (`OSWORLD_EVAL_MODEL_*`) — V2 tasks
  that need the LLM judge require running inside the official V2 harness.

External infra required before any real run on this machine (none of it is
vendored here): an OSWorld checkout (ideally the pinned V2 commit with
`uv sync`), a provider that can actually host the Ubuntu VM — VMware Fusion
with the official VM image, or a Docker host **with KVM** (macOS Docker/colima
has no KVM; OSWorld's docker provider targets Linux hosts) — plus provider
API keys in an isolated settings file. The preflight fails loudly listing
exactly which of these is missing.

## 6. Reporting rule

Report the mean of official `result.txt` scores over the attempted task set,
with the step budget, provider, model, checkout commit, and every scaffold
disclosure above. Runs without the official V2 checkpoint protocol are
final-state-only numbers; say so. Preflight-blocked and adapter-error tasks
stay in the denominator via `result_index.jsonl` (`status=blocked` /
`adapter_error`).

The task-acceptance panel is a scaffold axis of its own (2026-09-02): it runs the
configured `OUROBOROS_REVIEWER_SLOTS.triad` rows on their own deliveries (API packet,
configured-subagent native inspection episode, or agent session) instead of an
API-only projection. The runs recorded here used three API packet rows, so their
acceptance axis is unchanged; a run whose triad contains a retrieving row is not
comparable on that axis — declare the triad rows with the scaffold disclosures and do
not merge it with API-panel runs under the §7.9 dedup rule.

**Do NOT score by filtering `result_index.jsonl` rows on `status == "completed"`.**
`status` is a claim about the RECORD, not about the evaluation:

- `status="partially_published"` means the score was obtained and the official
  evaluation really ran, but at least one destination (the outcome sidecar, the
  manifest amendment, the ledger) could not be written. The status the run
  reached is preserved verbatim in `details.outcome_status`, and
  `details.publication_errors` lists the gap. Filtering on `status` silently
  drops these SCORED rows.
- `output_paths.task_outcome` MAY BE ABSENT. The pointer is emitted only when
  that write actually succeeded — a row naming a path that does not exist is
  worse than one naming none, because a reader cannot distinguish it from a
  file deleted later. The finalized attempt manifest follows the same rule.
- `runtime_outcome` (see §7.4) carries the RUNTIME's own terminal reason,
  independently of the adapter-stage `status`/`reason_code`. A cost-truncated
  task publishes `status="completed"`, `reason_code="official_evaluate"` and
  `runtime_outcome.reason_code="budget_exhausted"`, `truncated=true` — all
  three are true at once and the artefact must be read that way.

Score from `details.reward` / `official_eval_status` (which are deliberately
never demoted by a publication failure), and use `status` only to judge how
complete the record is.

A refused runtime attestation keeps its EVIDENCE: all three launchers catch
`RuntimeAttestationRefused` and persist the record it carries (the exact typed
reason plus `runtime_version`, `repo_head`, `repo_version`) under
`extra.runtime_attestation`, rather than only the string
`runtime_attestation_failed` — so a blocked row says WHICH runtime disagreed with
WHICH commit.

All three launchers ADMIT their run through the shared seams in
`common/manifests.py`: `admit_benchmark_run()` builds the manifest from pure
argument derivation, WRITES it, and only then does the provenance/clean-seed gate
enforce, so a dirty or git-identity-less checkout stops the run BEFORE the
preflight and the VM boot AND leaves a durable record of what was refused;
`finalize_run_manifest()` records the terminal `outcome` plus the process's real
`exit_code` on every exit path. Nothing touches the filesystem before admission.
The `--allow-dirty-seed` escape is recorded in the manifest
(`extra.allow_dirty_seed` and the `seed_gate` block); a run that used it is not
submittable.

When several run directories cover the same task (resumes, retries, or a future
laned run — the lane generator is NOT part of this release, see §7.9),
merge them with the pre-registered dedup rule in §7.9 — **first scored attempt
wins** — never by picking the best score.

## 7. cu_bridge runner: protocol deltas and disclosures

`run_cu_bridge_agent.py` is a SECOND runner with a different shape from the
step-loop `run_step_agent.py`. It reuses the official environment but changes
the control loop, so its numbers are NOT drop-in comparable to a step-loop or
leaderboard run without the disclosures below.

1. **Runner identity.** One persistent Ouroboros task per OSWorld task
   (`--memory-mode empty`, full within-task memory) drives the VM through the
   bundled `unix_computer_use` skill's `osworld_http` backend — the
   Terminal-Bench / Pointer shape. The step-loop runner instead has the host
   call `env.step` with Ouroboros as a stateless per-step action selector.
2. **Actions bypass `env.step` (same guest channel).** GUI actions are sent
   straight to the in-VM `/execute` pyautogui server — the SAME guest channel
   `env.step` uses — so guest mutation is identical, but the official
   `DesktopEnv.action_history` and `traj.jsonl` are NOT populated. `reset()` and
   `evaluate()` are the official ones. State evaluators depend only on final VM
   state, so scoring is state-correct; the one action-history dependency is the
   infeasible evaluator (below).
3. **Infeasible = final-answer convention → official `FAIL`.** When the agent's
   FINAL answer (`final_answer`/`result` field only, standalone line
   `TASK_INFEASIBLE`) declares infeasibility, the bridge emits an official
   `env.step("FAIL")` before `evaluate()`. OSWorld scores `infeasible` tasks 1.0
   iff the last recorded action is `FAIL`, so this matches the official
   semantics; a `FAIL` on a feasible task scores 0. Detection reads only the
   terminal answer (never intermediate reasoning) to avoid spurious flips.
4. **Budget is rounds + wall-clock + a per-task USD rail — and, since v6.84.0,
   ALSO a declared leaderboard step cap.** Before v6.84.0 there was no per-task
   STEP cap; `--max-steps` now declares one and the runtime enforces it (see
   4a-bis). The other three caps still apply, and the one that binds in
   practice is the third:
   - the bench server's `OUROBOROS_MAX_ROUNDS` (default 200);
   - `--task_timeout_sec` wall clock;
   - **the runtime's per-task USD reservation rail**
     (`OUROBOROS_PER_TASK_COST_USD`, enforced by
     `usage_accounting.reserve_attempt`: it refuses when
     `root_accounted + reservation_upper_bound > root_limit_usd`). The bound is
     a WORST-CASE estimate that grows with multimodal context, so on a
     screenshot-heavy OSWorld task it reaches the rail far below actual spend —
     in the v6.81.0 smoke it tripped a $6.00 rail at **$0.45 of actual spend**,
     stopping tasks at 13 and 22 rounds while `max_rounds_effective` reported
     200. An earlier version of this item claimed there was no other per-task
     cap; that was false, and it is exactly the kind of claim this file exists
     to prevent.

   A leaderboard "step" is one model turn (which may batch several pyautogui
   actions). CORRECTED in v6.84.0: an Ouroboros top-level policy round IS that
   turn — the official loop increments `step_idx` once per `agent.predict()` and
   executes every action that turn emitted. The earlier claim of
   non-equivalence compared a TURN against an ACTION and understated our budget
   by roughly 2.4x. `task_outcome.json`
   records `budget_counters` (`llm_rounds`, `screenshots`, `gui_action_calls`,
   `remote_exec_calls`) and `max_rounds_effective`; report these alongside any
   score. Since v6.81.1 the round is ALSO not comparable to earlier Ouroboros
   runs' rounds: screenshot images auto-attach in the same round, where every
   earlier run spent a second `view_image` round per observation (v6.81.0:
   3,830 of 16,367 rounds — ~21% of the budget — were that second round). When
   comparing across the boundary, compare `screenshots`/`gui_action_calls`
   counts, not raw `llm_rounds`. **A task the USD rail truncated is NOT a capability failure.** Read
   `runtime_outcome` in `task_outcome.json` / `result_index.jsonl`: it carries
   the runtime's own `reason_code`, a `truncated` flag and the `resource_limit`
   block. `truncated` is true for every code in
   `result_index.RUNTIME_TRUNCATION_REASON_CODES`, which is DERIVED from
   `ouroboros.outcomes.BEST_EFFORT_REASON_CODES` rather than restated — so the
   USD rail (`budget_exhausted`), the round cap (`round_limit`) and the
   loop-local deadline (`deadline_local`) are all covered, and a code the
   runtime never emits cannot be listed. Report truncated tasks separately from
   honest failures; `max_rounds_effective` alone will not tell you they
   happened. The current Verified leaderboard standard is 100 steps.
4a-ter. **Scaffold revision v6.86.0 — again a NEW series.** The working prompt gains
   an ATOMIC TASK CONTRACT: before the first mutating action the agent writes the
   task's obligations as a numbered checklist (object, required state with every
   stated qualifier, order/position, what must stay unchanged, where the result must
   persist), and before finishing it closes each item as observed-satisfied /
   not-verified / impossible, with at most one targeted repair. Plural instructions
   ("all", "each") still cover every matching element; the singular-referent rule
   applies only to a singular referent resolving to several candidates, and the
   contract is explicitly revisable on new observation. Three infeasibility shapes
   are named (discovery outside a stated means restriction; a named mode of operation
   the application does not ship; a mechanism whose trigger is narrower than the task
   states). The desktop environment's own documented configuration CLI
   (gsettings/dconf) is declared a legitimate surface — private application state
   (prefs.js, profile directories, document XML, credential stores) remains
   forbidden. The colour clause keeps the app's named palette entry and no longer
   claims the reference file was authored from that palette (false on at least one
   graded task, where the metric is distance from a pure primary).
   Harness: each `proxy:true` task gets its own sticky upstream session
   (`;sessid.<tag>` keyed on campaign root + example id, so two concurrent campaigns
   never share an exit), written to LANE-PRIVATE state and deleted after the task —
   never under `results/`, which is the published tree; and after the post-gate reset
   the runner probes whether the binaries the task's setup claims to install are
   actually present, recording the answer in the manifest, because upstream reports a
   guest command that failed as "executed successfully" and a vanished premise
   otherwise surfaces as an honest-but-scored-zero infeasible.
   Numbers from v6.86.0 MUST NOT be pooled with v6.81.x, v6.83.0 or v6.84.0.

4a-bis. **Scaffold revisions v6.83.0 and v6.84.0 — a NEW series, not comparable to
   earlier numbers.** v6.83.0 introduced a declared and enforced step budget (one step
   = one top-level policy turn, matching the official `predict() -> actions[]`
   boundary; 100 total = gate + worker + one terminal turn), residential-proxy routing
   for `proxy:true` tasks, and a package of working-prompt clauses. v6.84.0 revises
   that package on measured evidence from the v6.83.0 runs: the budget clause is
   restated in TURNS (the previous wording charged tool calls and produced 1.01 calls
   per turn across ~11k turns, i.e. the runs used about a third of their action
   budget); three clauses that agents demonstrably cited while losing are corrected
   (an explicit numeric value beats a preset, "already in the requested state" is
   judged from the stored value rather than the render, and ordinals count list
   entries only inside an actual list); the shell is admitted for file-level batch
   deliverables while application state stays GUI-only; verification requires an
   independent read-back; and the premise rubric gains a narrow display branch.
   Harness: the official evaluator now runs with the checkout root as CWD against an
   absolute per-campaign cache, the gate's unused turn reserve is returned to the
   worker, proxy routing is gated on a live probe, and a `proxy:true` task whose own
   trace shows an exhausted upstream and which scored a hard zero is published as an
   unscored infrastructure result rather than a capability zero (disclose the count).
   Numbers from v6.84.0 MUST NOT be pooled or averaged with v6.81.x or v6.83.0
   numbers, and a delta across that boundary is a scaffold delta, not a model delta.

4a. **Scaffold revision after the v6.81.0 run — disclose which revision produced a
   number.** (v6.81.1 adds a further prompt revision, disclosed here under the same
   rule: the working preamble gains an "ENVIRONMENT PITFALLS" section of task-general
   state rules — live-app in-memory copies must be reconciled after out-of-band edits,
   terminal tasks belong in the visible terminal, PIDs are resolved by exact
   executable, never by self-matching `-f` patterns — derived from graded-run
   forensics but deliberately phrased without any claim about what an evaluator
   inspects; the vision-loop instruction drops its mandatory `view_image` step because
   screenshots now auto-attach; and the premise-gate prompt becomes the structured
   rubric of §4c. Numbers from before and after this revision must not be pooled.) Pairwise trace forensics against a published verified run on the same 361
   tasks and the same model showed the two runs statistically indistinguishable on the
   333 feasible tasks (0.804 vs 0.808) with ~90% of the deficit concentrated on the 27
   tasks whose evaluator is `infeasible`. Reading our own traces there found a
   reproducible behaviour: the agent ran the correct read-only probe, obtained the
   correct negative result, and then MANUFACTURED the missing premise — copying a system
   wallpaper onto an empty Desktop and "adjusting" its own planted file, creating a
   same-named theme directory by copying a sibling, writing document internals the
   application cannot render — and reported success. The pre-existing prose rules did not
   hold: the preamble already demanded a feasibility check and already forbade
   behind-the-back mutation.

   Two defects in those rules were identified and corrected in the adapter:
   the feasibility rule enumerated only missing hardware / accounts / application
   features, which does not cover an absent object the task acts on; and the
   investigation budget ("no more than 2 calls before a real GUI action") applied to the
   premise check as well, so establishing feasibility competed with speed.

   The revised adapter (a) states the premise rule as "an essential PRE-EXISTING target
   or capability the task presupposes is absent", explicitly distinguished from something
   the task asks to create, from a detail mentioned only as motivation, and from a
   transient network/anti-bot block — the last of which is not hypothetical: the
   published verified run lost `82279c77` by declaring INFEASIBLE on a Cloudflare block
   for a task whose evaluator is `check_direct_json_object`; (b) exempts the premise
   check from the investigation budget; (c) forbids manufacturing the premise; (d)
   forbids searching the guest for grader code (a behaviour observed in the v6.81.0
   traces — nothing was found and nothing was exploited, and it is now prohibited rather
   than merely unsanctioned); (e) adds literal-criterion verification, environment
   anchoring and canonical-surface preference; and (f) populates
   `task_contract.acceptance_claims`, which was empty on all 361 tasks of the v6.81.0 run
   while the task-acceptance panel ran on 324 of them — the reviewer was being paid for
   and told nothing.

   None of this encodes a base rate of infeasible tasks, a task id, an application, or
   any property of an evaluator. **Numbers produced before and after this revision are
   from different scaffolds and MUST NOT be averaged, pooled, or presented as one series.**

4b. **`final_answer` in cu_bridge outcomes was null and is now populated.** §4 promises the
   terminal answer is captured so the audit trail never shows an empty answer for an agent
   that answered. On this runner it was never written: every v6.81.0 outcome carried
   `final_answer: null` while the text sat in the runtime result. Outcomes written before
   this fix under-report what the agent said; the reward figures are unaffected.

4c. **`--feasibility-gate` (opt-in, off by default) posts UP TO two tasks per
   example** — one when the gate ends the example as INFEASIBLE (no working task is
   created), two when the working phase runs. A
   read-only premise phase runs first, with the mutating GUI tools absent from its
   capability envelope. Its prompt is a STRUCTURED RUBRIC, not an example list — the
   v6.81.0 false kills all judged whether the outcome would be meaningful instead of
   whether the requested action is performable, and enumerating those cases as
   exceptions would be a keyword patch. The rubric walks: requested ACTION →
   pre-existing REFERENT it needs → does absence BLOCK that action → is the missing
   thing ACQUIRABLE by means the instruction does not forbid → does a "set X to Y"
   target merely STORE the value → unbound PLACEHOLDERS in the instruction itself.
   Verdict is the last line: INFEASIBLE / PROCEED / UNDETERMINED.

   A single verdict decides (v6.81.1). The v6.81.0 revision additionally ran a
   confirming CHALLENGER round; its own full-run ledger removed it — 20 invocations,
   0 feasible tasks saved, 1 officially-infeasible task lost, 215 worker rounds
   burned, and it CONFIRMED all four of the gate's false kills. An identical-prompt
   re-read produces correlated errors, not an independent check. Only an explicit
   INFEASIBLE ends the example — translated into the official `env.step("FAIL")`
   through the same single path an agent-declared infeasibility takes. Everything
   else (PROCEED, UNDETERMINED, an unparseable answer, a timeout whose cancel
   confirmed, a crashed phase, any exception) falls through to the full-capability
   phase, which keeps its own mid-task `TASK_INFEASIBLE` path. The single deliberate
   exception to fail-open is a premise round whose CANCEL DID NOT CONFIRM: a zombie
   premise session shares the lane's server and skill connection file, so the attempt
   aborts as a typed infra row (`blocked` / `gate_cancel_unconfirmed`, claim released)
   rather than let it act on the VM the worker is scored on.

   Disclosures that belong with any gated number. (i) The manifest reports
   `one_run_per_task: false`, `feasibility_gate_phase: true` and
   `feasibility_gate_challenger: false` (the v6.81.0 manifests say `true` — the flag
   exists so readers of both runs see the scaffold difference), and each outcome
   carries `infeasible_source` (`feasibility_gate` vs `agent_final_answer`) plus the
   phase's own rounds. (ii) The claim staleness bound is widened by ONE premise window
   (`_gate_claim_window_sec`); without that a gated holder consumes the margin the
   formula reserves for the unbounded `evaluate()`, and a second lane can reclaim and
   re-score a task still being worked. (iii) The phase closes the GUI vector only:
   `remote_exec` remains a general shell, read-only by instruction and not by
   enforcement — so `feasibility_gate.json` carries the round's full tool trace
   (verbatim args, not previews) for offline audit, and the working phase is
   re-reset afterwards.

   **Every reset republishes the VM endpoint**, and this is the load-bearing repair.
   `DockerProvider.revert_to_snapshot` stops the container and `start_emulator`
   REALLOCATES ports (`_get_available_port(5000)`), so the VM's address changes on every
   reset. The 2026-07-28 v1 smoke wrote the endpoint once, before the gate (measured:
   83/83 task dirs had `bridge.json` older than their gate record), so after the
   post-gate reset the working phase kept driving the PRE-GATE address — which, with 16
   lanes allocating from the same port range, another lane's container could already
   own. Feasible-control mean 0.737 -> 0.459, and the signature was unmistakable in the
   traces: workers reported an empty Desktop and missing task files, and in one case
   acted on a completely different task's presentation. Republishing after the post-gate
   reset removed the whole class (v2: 0 regressions vs 9 at the comparable stage).

   Both resets are additionally VERIFIED (`_reset_verified`) against a SECOND, distinct
   fail-open in OSWorld itself: `reset()` skips every setup step when the guest server
   misses its probe window, then logs "Environment setup complete." and returns a
   pristine VM with no exception. The runner asserts the machine-checkable postcondition
   (`is_environment_used` true whenever the task config is non-empty, plus a screenshot
   over the same HTTP path the agent's tools use), forces the snapshot revert before
   every retry (an unforced retry runs setup on top of the partial state), and turns
   exhaustion into a typed infra row (`reset_unverified`, reward `null`, claim released)
   — a setup the harness could not verify must never become a capability zero. Reported
   honestly: this second guard has not yet fired in production (v2: 24 post-gate resets,
   0 retries), so it is defence in depth, not the measured fix. It does close a real
   pre-existing flaw that affects UNGATED runs too, on their single reset.

   Expect roughly double the per-example warm-up and acceptance-review cost on
   examples whose working phase runs; a gate-killed example pays only the premise
   round (~4 LLM rounds median in the v6.81.0 run). The number that decides whether
   it is worth paying is the
   FALSE-INFEASIBLE rate on feasible tasks — a wrong verdict scores a hard zero — and
   it must be measured on a stratified control set of feasible tasks before the gate is
   used for a scored run, WITH the control classes interleaved in the task list (a
   blocked layout delivers the flattering class first and the deciding class last).

5. **Observation modality.** Screenshot-only by DEFAULT: `ax_tree` is disabled
   per task unless `--allow-a11y`. A run with `--allow-a11y` must be reported as
   "Additional a11y tree used: Yes" (the leaderboard separates Screenshot /
   A11y / Screenshot+A11y / SoM).
6. **Shell access.** `remote_exec` gives the agent a guest shell (within
   OSWorld's action space — arbitrary python/subprocess is allowed by the
   harness). Disclose "Additional coding-based action: Yes" (Pointer, the #1
   Verified agent, does the same). The GUI-first preamble is advisory only; for
   a strict GUI-only number, `remote_exec` would have to be hard-disabled, not
   just discouraged.
7. **Sudo password injection.** The VM `client_password` is placed in the
   prompt (`prompt.txt`). This is official practice — the OSWorld baseline
   prompts (`mm_agents/prompts.py`) say "My computer's password is
   '{CLIENT_PASSWORD}', feel free to use it when you need sudo rights", and some
   tasks are unsolvable without sudo. Keep run artifacts access-controlled since
   the password appears in `prompt.txt`.
8. **Proxy policy.** OSWorld-Verified flags 52 tasks `"proxy": true`
   (anti-crawling / regional access). CORRECTED in v6.84.0: config existence is
   no longer sufficient — the bridge probes the configured gateway with a live
   CONNECT for each proxy-flagged task and enables the pool only if that
   succeeds, because a config pointing at an exhausted account still exists on
   disk while answering 407, and a DEAD proxy scores worse than none (measured:
   chrome-with-dead-proxy 0.249 mean vs 0.679 for everything else). When no live
   route exists the task still RUNS, direct, and its outcome records
   `proxy_required`, `proxy_enabled` and `proxy_exhausted_in_trace`; the scoring
   report must disclose how many proxy-flagged tasks ran on the direct protocol,
   because that is a protocol difference, not a capability result. Tasks are
   never dropped from the denominator for it: the lane makes a single pass over
   the task list, so skipping an example deletes it from the campaign. For a clean campaign, either provide a valid
   proxy config or publish the "proxy unavailable" subset separately rather than
   counting those as model failures.
9. **Overlapping runs and the dedup rule (v6.76.0). PRE-REGISTERED — this
   rule is fixed here BEFORE any numbers are looked at.**
   - **MULTIPLE LANES ARE SUPPORTED; THERE IS NO MULTI-LANE LAUNCHER GENERATOR IN
     THIS RELEASE.** Overlapping runs are a supported configuration and the
     v6.76.0 smoke exercises it: several lanes run concurrently, each an
     operator-written script invoking `run_cu_bridge_agent.py --claim-dir <shared>`
     against its own isolated bench server, over a SHARED results tree. What was
     built during this phase and then EXTRACTED is the CONVENIENCE GENERATOR for
     those scripts (`gen_lanes.py`, `bind_lane_ports`, `lanes.json`); it is
     deferred to a later release. So nothing in this tree generates lane scripts,
     allocates lane ports or starts more than one bench server — the operator
     does that — and no claim is made about an automated lane topology.
   - The claim mechanism below is what makes overlapping runs safe, and it is NOT
     lane-specific: append-only reruns, resumes and retry passes over a SHARED
     results tree need the same "who owns this task" answer that two concurrent
     lanes do, and the dedup rule is what makes an overlay of several run
     directories reproducible.
   - **PER-ATTEMPT RECORDS, ONE CANONICAL RESULT.** The per-task run directory
     `<result_dir>/<domain>/<example_id>/` is keyed by the TASK, so overlapping
     attempts share it. Every ADMITTED attempt therefore writes its OWN
     admission and finalization record to
     `attempts/<attempt_id>/task_run_manifest.json`, and only the attempt
     HOLDING the claim writes the canonical artefacts in the run directory
     itself (`task.json`, `result.txt`, `task_outcome.json`,
     `task_run_manifest.json`). Without that split, two lanes overwrote each
     other's admission record before either had claimed the task, and the loser
     then finalized `skipped_in_flight` on top of the holder's still-running
     record — defeating both the claim's ownership contract and the append-only
     evidence contract. The shared `result_index.jsonl` ledger stays
     append-only, but it is NOT a per-attempt log: an attempt enters it only
     when it produces an OUTCOME (`task_outcome.json` + one ledger row, written
     together). An attempt that steps aside on a held or already-scored claim
     produces no outcome and therefore no row (see **Claims** below); an attempt
     blocked before the claim — a failed seed gate or runtime attestation — does
     produce an outcome, so its row exists and carries `claim_owner: false`.
     Each row names its `attempt_dir` and whether that attempt was the
     `claim_owner`, so a reader deduping by `instance_id` can tell the holder's
     row from a bystander's. **So an auditor reconstructing a run reads the
     `attempts/` subtree for everything that was TRIED and `result_index.jsonl`
     for everything that produced an outcome and therefore counts in the
     denominator** — the split is what keeps a losing lane out of the canonical
     record, not an omission in it.
   - **Claims.** Two attempts can never both take the same task. Before seeding the
     skill or booting a VM, `run_cu_bridge_agent.py --claim-dir` takes an exclusive
     O_EXCL lock (`ouroboros.platform_layer.acquire_exclusive_file_lock`) whose
     staleness bound is `task_timeout + 2 × startup_timeout + margin`. TWO
     startup windows, because a holder gets one for the `DesktopEnv` constructor
     and a fresh one for the reset-to-usable-screenshot loop; a one-window bound
     could expire while a lane was still legitimately working. `env.evaluate()`
     runs after all of those and is UNBOUNDED (upstream getters may fetch over
     the network), so the formula cannot cover it — that residual is what
     `--claim-margin-sec` (default 900s) is for; raise it for domains with slow
     evaluators. An attempt that finds the task claimed or
     already scored exits with code 4 without producing an outcome, and therefore
     writes NO ledger row: the owning attempt owns the denominator row. Its own
     admission record under `attempts/<attempt_id>/` is where that attempt is
     visible. The scored STATE is answered with a READ-ONLY probe
     BEFORE admission (`scored_claim_state`), so a late attempt leaves no footprint
     at all in the winner's shared per-task run directory, and again UNDER THE HELD
     LOCK inside `acquire_task_claim`: reading it only before waiting for
     the lock is a live TOCTOU hole, because a contender that started before the
     winner scored would acquire the lock afterwards and still be told `claimed`.
     The lock taken purely to look is handed straight back. That state is read from
     MARKERS and never from the lock, so a refusal on it never expires — the lock is
     deliberately reclaimable (`stale_sec` recovers a crashed holder's task) and a
     protection built on it would fail open once somebody waited long enough. No
     daemon, no registry, no lease.
   - **DEDUP RULE: FIRST SCORED ATTEMPT WINS.** Runs are append-only, so a task
     may legitimately appear in several run dirs (a crashed lane, a resume, a
     later retry pass). When merging/overlaying results, the authoritative row
     for a task is the EARLIEST attempt that produced an official
     `env.evaluate()` score — regardless of its value. Later attempts of an
     already-scored task are recorded but never counted, and a higher reward from
     a later attempt is NEVER preferred (that would be best-of-N under another
     name). Unscored attempts (adapter error, blocked preflight, crashed lane)
     do not consume the task: only a scored attempt leaves the permanent
     `<claim>/<key>.scored` marker, so a retry lane may take an unscored task
     again. That marker is the RULE'S AUTHORITY, not an optimisation, so it is
     fail-closed and durable: `mark_task_scored()` fsyncs it immediately after
     `env.evaluate()` and BEFORE the reward is written to `result.txt` or any
     outcome/ledger artefact, a write failure raises instead of being swallowed,
     and a scored claim is never released while its marker is unconfirmed. The
     only crash orderings reachable are therefore "marker, no result" (a later
     lane steps aside; the missing row is visible in the denominator) and
     "no marker, no result" (a later lane legitimately retries) — never
     "result without marker", which is what made a lane rerun a task that already
     had an official score. If the canonical marker cannot be persisted, the task is
     SCORED BUT UNMARKED, and that state is itself recorded DURABLY:
     `mark_task_scored()` fsyncs `<claim>/<key>.scored_unconfirmed` (one further
     path, NOT a further layer of best-effort) and `scored_claim_state` refuses the
     task on it with its own typed reason `scored_unconfirmed`, REGARDLESS of
     staleness — so the refusal is permanent and visible to an operator instead of
     the task silently becoming claimable again. The runner also retains its
     in-flight lock as interim cover, reports the official reward with
     `reason_code=claim_marker_not_durable` and `claim_lock_retained`, and exits 2.
     Retaining the lock alone was NOT enough: `stale_sec` reclaims it by design, so a
     lock-only protection merely delayed the rerun of an already-scored task.
     If even the fallback marker cannot be written, NOTHING on disk records the
     score and there is no protection left to promise: the runner refuses loudly
     with `reason_code=claim_state_unrecoverable` and exit 3, stating that the claim
     directory is unusable and further tasks must not be run against it. Such a task
     must be reconciled by hand before the claim dir is reused; a
     `.scored_unconfirmed` marker is cleared deliberately, never by expiry.
     The claim directory itself is confined: `--claim-dir` is resolved through the
     same `assert_outside_repo`/`live_data_roots()` boundary as every benchmark
     output root before anything is created, so lock and marker files can never
     land in a repository checkout or the owner's live runtime data. The authority
     is the EXECUTION checkout (`--repo-dir`, the tree the manifest attests) and
     the launcher's own checkout — both, not either. Confining against the
     launcher's location alone let `--repo-dir /other/bench-clone --claim-dir
     /other/bench-clone/.claims` write lock and marker state into the very seed
     whose cleanliness the gate was about to attest.
   - **THE INTERRUPT WINDOW IS CLOSED; THE `SIGKILL` WINDOW IS NOT — say so when
     reporting a hard-killed run.**
     An interrupt (`KeyboardInterrupt`/`SystemExit`, which are `BaseException` and
     bypass an `except Exception`) between `env.evaluate()` and the marker is
     handled DURABLY: `<claim>/<key>.scored_unconfirmed` is fsync'd before the
     interrupt is re-raised, so the task is refused permanently and the operator
     sees why. Retaining the in-flight lock — which is all this path used to do —
     was a protection with a countdown on it, because `stale_sec` reclaims that
     lock BY DESIGN: once the staleness bound passed, the next attempt was handed
     an ALREADY-EVALUATED task and the score was counted twice. The lock is still
     retained as interim cover, but the refusal now comes from a marker that
     cannot expire.
     A `SIGKILL` in that window is NOT handled and cannot be — no handler runs by
     definition. That window is exactly: `env.evaluate()` has returned a score and
     neither `mark_task_scored()` nor the interrupt path has completed its one
     fsync'd atomic write.
     What the `SIGKILL` window costs is bounded, and it is NOT a wrong number. The
     marker is written BEFORE `result.txt`, `task_outcome.json` and the ledger row,
     so a kill in that window leaves NO record of the score anywhere: there is no
     row for an overlay to select, and a later attempt retrying the task is correct
     rather than a rerun of a counted score. The cost is one lost evaluation. (This
     is exactly why the INTERRUPT case is different and had to be closed: there the
     process does get to run code, and leaving only the expirable lock behind meant
     a score that WAS durably recorded could be counted a second time.)
     We deliberately did NOT close the `SIGKILL` window with an intent marker before
     `env.evaluate()`. That would block staleness reclaim for the whole evaluation
     — which is UNBOUNDED (upstream getters fetch over the network) — so every hard
     kill during evaluation (OOM killer, host reboot, an operator killing a hung
     evaluator) would leave a NEVER-SCORED task permanently refused and needing
     manual clearing. That trades a narrow benign window for a broad harmful one.
     **After a hard kill, check the claim dir against the results tree:** a task
     with a `.scored`/`.scored_unconfirmed` marker but no `result.txt` or ledger row
     was scored and lost its projection — reconcile it or report the missing row in
     the denominator; a task with neither marker and no result is safely retryable.
   - **VM boot failures are retried; the teardown is belt-and-braces.** The
     benefit claimed here is the RETRY: previously only `env.reset` was retried,
     so a single transient `DesktopEnv.__init__` failure (a lost port-allocation
     lock race, a slow image load) burned the whole task.
     `run_step_agent.construct_desktop_env()` now retries the constructor within
     the startup window. It also tears down every failed attempt (`env.close()` →
     `provider.stop_emulator()`) because a raise inside `__init__` discards the
     half-built object, leaving whatever `_start_emulator()` had already started
     unreachable. That teardown is a PRECAUTION, not a fix for measured debris —
     no run here has been shown to accumulate leaked containers.
   - **Provider lock.** Upstream holds a single GLOBAL
     `/tmp/docker_port_allocation.lck` across port allocation AND
     `containers.run` with a 10-second `LOCK_TIMEOUT`, so any concurrent docker
     container start loses the race and raises `filelock.Timeout` before its agent
     starts. The lockfile is host-wide, not run-wide: on this SHARED daemon the
     contender can be a resume pass, another of our runs, or another user's
     container entirely — the lane generator is deferred out of this release, but
     the contention is not lane-specific and does not go away with it. Both halves
     of the fix are applied: the tracked operator patch
     `operator_patches/osworld_docker_lock_timeout.v6760.patch` raises it to 60s,
     and our constructor retry above absorbs the residual races. Report runs made
     without the patch as such — the retry alone means more lost boots, not wrong
     scores.
10. **Dataset pin + self-reported vs verified.** The manifest records the
   checkout `variant` (V1-Verified: 369 tasks, 361 without the 8 Google-Drive
   tasks; OSWorld-V2: 108 long-horizon tasks) and git commit under
   `osworld_checkout`. Locally produced numbers are self-reported by definition;
   the Verified leaderboard requires the maintainers to run the agent code (or a
   trusted institution to share trajectories + monitoring). The bridge does not
   emit official `traj.jsonl`, so a Verified submission needs the official
   step-loop path or maintainer-side execution.
