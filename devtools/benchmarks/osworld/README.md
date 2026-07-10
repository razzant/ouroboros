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
  runnable-vs-skeleton status, and the cu_bridge final_answer lesson.
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
  adapter; the skeleton remains a stricter installed-agent preflight path.

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

Example smoke:

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
