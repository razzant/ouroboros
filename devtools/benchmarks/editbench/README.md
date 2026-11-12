# editbench — file-editing tool benchmark

> **Outcome note.** This benchmark drove the final toolset decision:
> `apply_patch` and `edit_batch` were adopted; `edit_sketch` (fast-apply via
> the LIGHT model) was evaluated here and REMOVED — the sketch/apply split
> never beat direct tools on cost or robustness. Its verification rails
> (unified diff in the result, pre-write syntax check) moved into
> `write_file`. The edit_sketch rows below are the historical evaluation data.

Measures Ouroboros' file-editing tools on a trap-laden multi-file rename task:
correctness, tokens, wall time, tool-call error rates, and which tools the agent
picks when free to choose.

## Task

`fixtures/toyproj/` is an 8-file Python project. The task: rename every
**module-level** function named exactly `ddd` to `aaa` (def lines, calls,
imports, aliases, qualified refs) while leaving everything else byte-identical.

Traps a naive replace-all falls into:

| Trap | Where |
|---|---|
| method `Widget.ddd` must keep its name (and `w.ddd()` call sites) | models.py |
| `w.ddd() + ddd() + core.ddd(2)` — rename 2 of 3 refs in one line | models.py |
| `ddd_helper`, `addd`, `dddx` identifiers | core/utils/legacy.py |
| local variable `ddd` (not a function) | utils.py `local_shadow` |
| string literals `"ddd"`, dict key, `"ddd mode"` | config.py |
| comments and docstrings mentioning ddd | core/report/utils.py |
| README.md mentions ddd and must not be touched | README.md |

Ground truth: `fixtures/expected/` (both versions pass `python main.py` → `OK`).
Grading (`checker.py`): per-file text equality (trailing whitespace normalized)
+ behavior check. `pass` requires 8/8 files AND behavior OK.

## Configs

Each config restricts the editing toolset via `disabled_tools`; shell/process,
web, delegation, review tools are disabled in EVERY config so the agent cannot
bypass the editing tool under test (e.g. `sed` via run_command).
`OUROBOROS_TASK_REVIEW_MODE=off`, `OUROBOROS_SAFETY_MODE=off` keep token counts
about editing work only. Its isolated settings also write the canonical explicit
`enabled=false` Available-subagents value with one API row naming the effective
Main model. Delegation is outside the measured editing surface and cannot reappear
through install defaults or a legacy Heavy value, while the manifest still records
the exact model the disabled route would have used. With no `--model`, Main is
materialized from the live benchmark seed (then environment/shipped default).

- `write_file_only`, `edit_text_only`, `apply_patch_only`, `edit_batch_only`
  — exactly one editing tool available (`edit_sketch_only` was removed with the
  tool; its rows below are historical data, not a config you can run today);
- `default` — the historical toolset (write_file + edit_text);
- `full` — all editing tools available (agent's free choice).

## Run

Fixed tasks, today's tools: a task never changes under its version label, and
every run measures the editing tools of the CURRENT working tree.

```bash
# Once per checkout, for t2-t5 (free, no model, no git history needed):
python devtools/benchmarks/editbench/make_fixtures_v2.py
# Paid: real model calls on an isolated server.
python devtools/benchmarks/editbench/run_editbench.py --tasks all --configs all --runs 3 --parallel 2 --allow-dirty-seed
```

`--tasks` defaults to `t1_rename`. `--allow-dirty-seed` is required, and
recorded, whenever the tree has uncommitted changes, which is editbench's normal
mode: it exists to measure editing-tool changes before they are committed.

Spawns one isolated Ouroboros server from the CURRENT working tree (clone +
`git diff HEAD` + untracked files), submits tasks over HTTP, mines
`llm_usage` rows (events.jsonl, deduped by ledger attempt ids) and `tool_call`
rows (tools.jsonl) from the isolated data root. Results: `runs.json` +
`summary.json` + `run_manifest.json` (see `--out`).

Before the clone, the server or any model call, every selected task is checked:
its expected tree must pass its own oracle, its untouched workspace must not,
and a t2-t5 task must match the pinned workload byte for byte. A failure, a
missing or half-built fixture tree included, ends the run as a recorded refusal
in `run_manifest.json` (`extra.refusal.stage` is `fixture_preflight`, exit code
2) that lists each problem. An admitted run writes `workloads` to the same
manifest before it executes anything (recovery checkpoint `fixtures_validated`),
so a run that dies mid-way still names its tasks. For t2-t5 that is the workload
version and a digest of the task's inputs and of the shared grader `checker.py`,
whose sha256 is listed beside it. t1 is named by its tracked fixtures at the
seed commit the manifest records.

## Results (2026-08-02, main model x-ai/grok-4.5, 3 runs/config)

All configs passed 8/8 files + behavior on every run — the traps did not catch
the main model regardless of tool. The discriminating signals were mechanism
overhead and format robustness.

| config | pass | avg wall s | avg completion tok | avg LLM calls | edit calls (3 runs) | edit errors |
|---|---|---|---|---|---|---|
| edit_batch_only | 3/3 | 66.3 | 2432 | 5.0 | 3 (1/run) | 0 |
| edit_text_only | 3/3 | 64.3 | 2542 | 5.7 | 19 | 0 |
| write_file_only | 3/3 | 69.3 | 2594 | 5.0 | 18 | 0 |
| apply_patch_only (v2) | 3/3 | 70.4 | 2438 | 6.7 | 10 | 0 |
| edit_sketch_only (v2) | 3/3 | 67.3 | 2351 | 5.0 | 18 (6/run) | 0 |
| default (write_file+edit_text) | 3/3 | 75.3 | 2541 | 5.0 | 18 edit_text | 0 |
| full (free choice) | 3/3 | 75.3 | 2542 | 5.7 | 13 edit_text + 1 edit_batch | 0 |

v1 (before robustness fixes) recorded the two failure modes the fixes removed:
- apply_patch v1: 3 errors / 11 calls — the model decorated headers as
  `*** Update File: x ***`; the parser now tolerates trailing asterisks.
- edit_sketch v1: 18 errors / 36 calls, 123 s and 6036 completion tokens per
  run — on these tiny files the model naturally sends complete-file sketches
  without `... existing code ...` markers, which were rejected. Marker-less
  sketches are now verified and written directly (no apply model call).

Takeaways: `edit_batch` (counted replace-all) finishes the whole task in ONE
tool call; format-tolerant parsing matters more than clever matching for
apply_patch; the free-choice agent prefers edit_text, so tool descriptions are
the steering surface if a different mix is desired.

## v2: real-file tasks (fixed workload v2.1)

| task | file(s) | shape |
|---|---|---|
| t2_surgical | review_state.py (1691 lines) | 2 helper renames + 2 constant bumps = 11 changed lines |
| t3_blocks | shell_parse.py (460 lines) | replace 3 whole functions with code given verbatim in the prompt |
| t4_move | shell_parse.py + git_shell_policy.py | move a function across files (delete + insert renamed + fix import/call site) |
| t5_overhaul | provider_models.py (423 lines) | flip 208 double-quoted literals to single quotes (119 lines change, AST-identical) |

The four tasks are pinned data, independent of today's product source. Their
seeds ship in `seeds_v2/` as `<name>.py.txt`: the exact bytes of the `ouroboros/`
files of main on 2026-08-02 (commit `1b7e097e1`), the date of the August tables
below, whose files reproduce every count those tables document. The `.txt`
suffix keeps them out of imports, linters and the module size gate; the 1691-line
seed is the t2 workload itself. Each seed's sha256 is checked on every build, and
its git blob id is kept as provenance: in a full clone `git cat-file blob <id>`
prints the same bytes, but nothing reads history, so a shallow clone or a source
archive materializes the same workload. `make_fixtures_v2.py` derives the
expected trees and pins the prompt, the check script and every materialized file
by sha256 (`PINS`); together they form workload `v2.1` (`WORKLOAD_VERSION`).
These files are benchmark inputs and are never imported by the product.

The generator builds into `fixtures_v2.staging-*`, validates the whole set (the
pins hold, each expected tree passes its oracle, no untouched workspace does) and
only then swaps it into `fixtures_v2/`, keeping the replaced set as
`fixtures_v2.previous/`. A failing build publishes nothing and leaves the
published set as it was; rerunning on a valid set changes nothing. The swap is
two renames, not an atomic exchange: a crash between them leaves no
`fixtures_v2/` and the replaced set in `fixtures_v2.previous/`. The runner then
refuses with a recorded `fixture_preflight`, and rerunning the generator
publishes again and keeps that set. All three directories are git-ignored data.

Changing a prompt, a check, a seed or the derivation creates a new workload: bump
`WORKLOAD_VERSION`, re-pin from the digests the failing build reports, and keep
results of different versions apart.

### Why v2.1 exists

Until v2.1 the generator re-derived the tasks from the CURRENT `ouroboros/`
files and deleted the previous fixtures before checking anything. Once those
modules moved on, every build failed (today's `review_state.py` has one
`_stable_digest`, not six, and `shell_parse.py` no longer defines `_SHELLS`, which
the t3 replacement code calls) and left a partial `fixtures_v2/`: a t2 workspace
without its expected tree, which the runner's workspace-only check admitted to a
paid run. v2.1 reads the August seeds instead. On those seeds the old derivation
and the new one produce byte-identical files under Python 3.10. Under Python
3.12+, which tokenizes an f-string in pieces, the old t5 oracle left
`f"{provider}::"` double-quoted although the prompt asks for it to be flipped,
so a correct answer would have been graded as a failure. v2.1 reads an f-string
whole on every interpreter and gives the August counts (208 literals, 119 lines).

### v2 results (2026-08-02, grok-4.5, 3 runs/config, 63/63 pass)

These and the weak-model tables below are historical. They were measured with the
August tools (including the since-removed edit_sketch), models and runtime, on
fixtures the old generator derived from the working tree of that day. Every count
documented here reproduces on v2.1, but the August fixture bytes were never
recorded, so read the tables as context, not as a baseline for today's tools;
compare new runs with runs of the same workload version.

Avg per run; errors are failed edit-tool calls across the 3 runs.

| task | config | wall s | completion tok | cost $ | edit calls | errors |
|---|---|---|---|---|---|---|
| t2_surgical | apply_patch | 57 | 1427 | 0.129 | 1.3 | 0 |
| t2_surgical | edit_batch | 54 | 1180 | 0.140 | 1.3 | 1 |
| t2_surgical | edit_text | 59 | 1856 | 0.181 | 9.7 | 0 |
| t2_surgical | edit_sketch | **229** | **20972** | **0.352** | 1 | 0 |
| t2_surgical | full → chose edit_batch | **48** | **1065** | 0.174 | 1 | 0 |
| t3_blocks | apply_patch | 47 | 1208 | 0.115 | 1 | 0 |
| t3_blocks | edit_text | 52 | 1536 | 0.122 | 3 | 0 |
| t3_blocks | edit_sketch | 99 | 6549 | 0.155 | 1 | 0 |
| t3_blocks | full → chose edit_text | 53 | 1372 | 0.117 | 2 | 0 |
| t4_move | edit_text | 63 | 2519 | 0.110 | 4 | 0 |
| t4_move | apply_patch | 62 | 2395 | 0.131 | 1 | 0 |
| t4_move | edit_sketch | 161 | 12107 | 0.232 | 2 | 0 |
| t4_move | full → chose edit_text | 62 | 2584 | 0.121 | 4 | 0 |
| t5_overhaul | edit_sketch (marker-less → direct write) | **125** | 6965 | **0.162** | 1 | 0 |
| t5_overhaul | write_file | 130 | 7364 | 0.196 | 1 | 0 |
| t5_overhaul | edit_text | 166 | 11373 | 0.261 | 8.3 | 0 |
| t5_overhaul | edit_batch | 178 | 11488 | 0.278 | 4.7 | 0 |
| t5_overhaul | apply_patch | **196** | 13344 | **0.309** | 13.3 | 2 |
| t5_overhaul | full → chose write_file | 134 | 8111 | 0.176 | 1 | 0 |

### v2 takeaways

- **No universal winner — the profile follows task shape.** Scattered surgery in a
  big file: apply_patch/edit_batch (edit_text needs 29 calls and re-reads).
  Whole-block replacement: edit_text/apply_patch tie. Near-full rewrite:
  write_file / marker-less edit_sketch; apply_patch is the WORST there (patch ≈
  file size split over 40 calls, 491k prompt tokens).
- **edit_sketch inverts**: worst for surgical edits in big files (the light model
  regenerates all 1700 lines: 21k completion tokens, 4× wall) — best for
  near-full rewrites (marker-less sketch = verified direct write).
- **The free-choice agent picked the right tool on every task** (edit_batch on
  t2, edit_text on t3/t4, write_file on t5) and was never worst, twice fastest.
  Tool diversity + honest descriptions is enough; no routing logic needed.
- Correctness was never the discriminator (63/63): the value of the new tools is
  fewer round trips, fewer retries, and the counted/atomic safety rails.

## Weak-model run (google/gemini-3.6-flash, t1+t2, old-vs-new configs, 2 runs each)

Pinned via `--model` (main lane + fallbacks). 16/16 pass — correctness did not
break even on the weak model at this task scale; the errors that did happen were
all caught atomically by the tools. What changed is the SIZE of the efficiency gap:

| task | config | wall s | prompt tok | completion tok | cost $ | edit calls | errors |
|---|---|---|---|---|---|---|---|
| t1 | edit_batch | **292** | **1.40M** | 10.3k | **0.70** | 1 | 0 |
| t1 | apply_patch | 349 | 1.70M | 11.6k | 0.80 | 1.5 | 1 |
| t1 | edit_text | 482 | 2.55M | 11.2k | 1.04 | 11 | 0 |
| t1 | write_file | 540 | 2.68M | 15.8k | 1.12 | 6 | 0 |
| t2 | edit_batch | **89** | **0.56M** | **2.7k** | **0.44** | 1 | 0 |
| t2 | edit_text | 148 | 1.04M | 4.1k | 0.58 | 9 | 0 |
| t2 | apply_patch | 229 | 1.28M | 11.0k | 0.68 | 3 | **4** |
| t2 | write_file | 224 | 1.23M | **38.5k** | 0.96 | 1.5 | 0 |

- The old toolset's cost/latency penalty roughly DOUBLES on the weak model:
  edit_batch is 1.5× cheaper than edit_text and 2.2-2.5× cheaper/faster than
  write_file (which finally shows the asymptotic completion-token pain: 38k
  tokens to rewrite 1691 lines — byte-perfect, but slow and expensive).
- apply_patch degrades on the weak model: flash writes sloppier patches (4
  format/context errors in one t2 run, all rejected atomically, retried to
  success). Its value is model-quality-dependent; edit_batch is robust everywhere.
- Flash is ~7× more expensive per run than grok-4.5 here despite cheaper
  tokens — weak models iterate far more (38-66 LLM calls vs 5) and cache poorly.

## Caveats

- Toy files are ~20 lines, so `write_file` (full rewrite) is barely penalized on
  completion tokens; on real-sized files the gap grows with file size. The
  benchmark discriminates mechanism overhead (round trips, retries, error
  rates), not asymptotic token cost.
- Prompt tokens are dominated by the system prompt (~30k/call, mostly
  prompt-cached); the discriminating signals are completion tokens, LLM call
  count, wall time, and edit-tool error counts.
- One grading run == one agent task; `--runs 3` per config for variance.
