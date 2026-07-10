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
- Single-model: solver/review slots all point at the same model
  (`settings_base.json`, secrets blank; fill keys at run time, never commit).
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
4. **Budget is rounds + wall-clock, NOT leaderboard steps.** There is no
   per-task step cap; the budget is the bench server's `OUROBOROS_MAX_ROUNDS`
   (default 200) plus `--task_timeout_sec`. A leaderboard "step" is one model
   turn (which may batch several pyautogui actions); an Ouroboros round is not
   step-equivalent. `task_outcome.json` records `budget_counters`
   (`llm_rounds`, `screenshots`, `gui_action_calls`, `remote_exec_calls`) and
   `max_rounds_effective`; report these alongside any score. The current
   Verified leaderboard standard is 100 steps.
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
   (anti-crawling / regional access). The bridge enables the proxy pool only
   when a proxy config file exists; otherwise a proxy task runs WITHOUT proxy
   and may fail on bot-detection. For a clean campaign, either provide a valid
   proxy config or publish the "proxy unavailable" subset separately rather than
   counting those as model failures.
9. **Dataset pin + self-reported vs verified.** The manifest records the
   checkout `variant` (V1-Verified: 369 tasks, 361 without the 8 Google-Drive
   tasks; OSWorld-V2: 108 long-horizon tasks) and git commit under
   `osworld_checkout`. Locally produced numbers are self-reported by definition;
   the Verified leaderboard requires the maintainers to run the agent code (or a
   trusted institution to share trajectories + monitoring). The bridge does not
   emit official `traj.jsonl`, so a Verified submission needs the official
   step-loop path or maintainer-side execution.
