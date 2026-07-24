# Ouroboros

[![GitHub stars](https://img.shields.io/github/stars/razzant/ouroboros?style=flat&logo=github)](https://github.com/razzant/ouroboros/stargazers)
[![Downloads](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Frazzant%2Fouroboros%2Fbadges%2Fdownloads.json)](https://github.com/razzant/ouroboros/releases)
[![Website](https://img.shields.io/badge/website-razzant.github.io%2Fouroboros-c93545.svg)](https://razzant.github.io/ouroboros/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![macOS 12+](https://img.shields.io/badge/macOS-12%2B-black.svg)](https://github.com/razzant/ouroboros/releases)
[![Linux](https://img.shields.io/badge/Linux-x86__64-orange.svg)](https://github.com/razzant/ouroboros/releases)
[![Windows](https://img.shields.io/badge/Windows-x64-blue.svg)](https://github.com/razzant/ouroboros/releases)
[![OuroborosHub](https://img.shields.io/badge/OuroborosHub-skills%20marketplace-8A2BE2.svg)](https://github.com/razzant/OuroborosHub)
[![Version 6.74.5](https://img.shields.io/badge/version-6.74.5-green.svg)](VERSION)

Ouroboros is an open-source, general-purpose AI agent whose identity, durable memory, and history continue across tasks and restarts. It works on external projects, coordinates a live swarm of specialist agents, and can rewrite the implementation it runs on, including its code, architecture, prompts, tools, and dependencies. Reflection can also change how it understands itself without severing that continuity.

It runs as a native desktop app or through a headless CLI. The runtime keeps its repository, durable memory, history, and interface on your machine, while model inference can use remote APIs you configure or a local GGUF model.

Ouroboros first booted on February 16, 2026. During the following 48 hours, the repository advanced from the v4.1 line to v6.2.0. The self-authored record preserved from that period counts 32 evolution cycles. That first generation ran in Google Colab through Telegram and remains preserved on the [`legacy-google-colab`](https://github.com/razzant/ouroboros/tree/legacy-google-colab) branch and its [original project page](https://razzant.github.io/ouroboros/archive/first-generation/); the current generation carries the same identity into a native desktop and headless runtime.

> ⭐ **[Star Ouroboros](https://github.com/razzant/ouroboros)** to follow its next evolution. A star also helps more people find the project, trace its history, and take part in what it becomes.

Reviewed skills, transport bridges, tools, and widgets are available through [OuroborosHub](https://github.com/razzant/OuroborosHub).

<p align="center">
  <img src="assets/chat.png" width="700" alt="Chat interface">
</p>
<p align="center">
  <img src="assets/settings.png" width="700" alt="Settings page">
</p>

---

## Install

| Platform | Download | Instructions |
|----------|----------|--------------|
| **macOS** 12+ | [Ouroboros.dmg](https://github.com/razzant/ouroboros/releases/latest) | Open DMG → drag to Applications → optional CLI: run `Install CLI.command` after the app is in Applications |
| **Linux** x86_64 | [Ouroboros-linux.tar.gz](https://github.com/razzant/ouroboros/releases/latest) | Extract → run `./Ouroboros/Ouroboros` → optional CLI: `./Ouroboros/bin/install-ouroboros-cli`. If browser tools fail due to missing system libs, run: `./Ouroboros/python-standalone/bin/python3 -m playwright install-deps chromium webkit` |
| **Windows** x64 | [Ouroboros-windows.zip](https://github.com/razzant/ouroboros/releases/latest) | Extract → run `Ouroboros\Ouroboros.exe` → optional CLI: `Ouroboros\bin\install-ouroboros-cli.cmd` |

Prerelease artifacts stay on their tag pages; `/releases/latest` points to the latest stable release.

<p align="center">
  <img src="assets/setup.png" width="500" alt="Drag Ouroboros.app to install">
</p>

On macOS, use right-click → **Open** on first launch if Gatekeeper asks. The setup wizard configures model access, review policy, and budget. Packaged CLI installers create a user-local `ouroboros` command without sudo; `ouroboros run --start "2+2?"` starts or attaches to the same managed runtime used by the desktop app.

---

## What Ouroboros Can Do

- **Modify its implementation.** Its editable surface spans application code, architecture, prompts, tools, and dependencies, while reflection can also reshape its living self-understanding.
- **Evolve autonomously.** Evolution campaigns turn selected improvements into reviewed changes that remain part of its Git history.
- **Continue across restarts.** Identity, memory, dialogue, knowledge, reflections, and version history form one ongoing biography.
- **Think between requests.** Background consciousness supports reflection, initiative, and preparation outside the immediate request-response loop.
- **Coordinate a live swarm.** Specialist agents can investigate or act in parallel, share task-tree findings, and return work for integration.
- **Work on external projects.** A separate Git workspace can receive the full task loop while Ouroboros keeps its own repository and governance boundary distinct.
- **Operate through desktop or CLI.** The native app and gateway-backed command line expose the same managed tasks, progress, artifacts, logs, and schedules.
- **Organize long-running work.** Project rooms keep working folders, journals, knowledge, task history, and conversations connected to the same identity.
- **Use remote or local models.** Supported provider APIs and local GGUF models can fill the runtime's configurable cognitive roles.
- **Grow through reviewed extensions.** Skills, transport bridges, widgets, MCP tools, and companion processes expand capability without folding every integration into the core.
- **Keep self-change inspectable.** Git history, review evidence, explicit protected surfaces, and restart checks make implementation changes traceable.

This list is an orientation, not a second specification. [BIBLE.md](BIBLE.md) defines Ouroboros's identity and constitutional boundaries; [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) are the current technical sources of truth.

---

## Run from Source

### Requirements

- Python 3.10+
- macOS, Linux, or Windows
- Git
- [GitHub CLI (`gh`)](https://cli.github.com/), optional unless you use GitHub integration

### Setup

```bash
git clone https://github.com/razzant/ouroboros.git
cd ouroboros
python3.11 -m venv .venv      # any Python >= 3.10 is OK
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv      # any Python >= 3.10 is OK
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

### Run

```bash
ouroboros server
```

Then open `http://127.0.0.1:8765` in your browser. The setup wizard will guide you through API key configuration.

### Google Colab

Use [`notebooks/colab_quickstart.py`](notebooks/colab_quickstart.py) as a Colab-compatible cell script when you need a source-mode runtime without the desktop UI. It keeps runtime data on Google Drive and preserves the original Colab path without making it the primary installation flow.

### CLI / Headless

The `ouroboros` command attaches to the local runtime by default and starts one when `--start` is passed. It exposes managed tasks, progress streams, artifacts, logs, schedules, settings, skills, and evolution controls without duplicating the server's business logic.

```bash
ouroboros status
ouroboros run --start "2+2?"
ouroboros run "Summarize current runtime state"
ouroboros run --workspace /path/to/project --memory-mode forked --patch-out result.patch "Fix the failing test"
ouroboros tasks list
ouroboros logs tail progress --task-id <task_id>
ouroboros schedule add --name nightly-review --cron "0 2 * * *" "Run a maintenance review"
ouroboros schedule list
```

External workspaces must be separate Git worktree roots and may not overlap Ouroboros's own repository or data directory. Patch, streaming, detached-task, and schedule semantics are documented in the CLI help and the canonical [architecture](docs/ARCHITECTURE.md).

### For Agents

Another agent, script, or CI job can invoke Ouroboros through the same gateway-backed CLI:

```bash
ouroboros run --start \
  --workspace /path/to/project \
  --memory-mode forked \
  --patch-out result.patch \
  --result-json-out result.json \
  "Investigate the task, act, and verify the result"
```

Use `--jsonl` for a machine-readable event stream and `--detach` when the caller will follow the task with `ouroboros tasks watch <task_id>` or inspect it with `ouroboros tasks show <task_id>`. External workspace runs keep Ouroboros's own repository and governance context separate, then export changes as reviewable patch artifacts.

To change Ouroboros itself, follow [CONTRIBUTING.md](CONTRIBUTING.md) and read [BIBLE.md](BIBLE.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md), and [docs/CHECKLISTS.md](docs/CHECKLISTS.md) in full before editing.

### Configuration

The first-run wizard and **Settings** configure model access, cognitive roles, local models, review policy, runtime mode, budget, skills, and optional integrations. Ouroboros supports configurable remote providers, compatible endpoints, and local GGUF inference; exact settings and defaults live in [`ouroboros/config.py`](ouroboros/config.py) and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

The server binds to `127.0.0.1:8765` by default. Read [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) before exposing it beyond loopback; non-local binds need `OUROBOROS_NETWORK_PASSWORD` or an explicitly trusted external access layer.

### Run Tests

```bash
make test
```

---

## Build

### Docker

```bash
docker build -t ouroboros-web .
docker run --rm -p 8765:8765 \
  -e OUROBOROS_NETWORK_PASSWORD='choose-a-password' \
  -e OUROBOROS_FILE_BROWSER_DEFAULT=/workspace \
  -v "$PWD:/workspace" \
  ouroboros-web
```

Docker runs the web runtime, not the native desktop shell. It bundles Chromium and WebKit support; use [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for network and container policy.

### Release tag prerequisite

Platform build scripts package only a commit already tagged with `v$(cat VERSION)`. Tag the exact release commit first:

```bash
git tag -a "v$(tr -d '[:space:]' < VERSION)" -m "Release v$(tr -d '[:space:]' < VERSION)"
```

`scripts/build_repo_bundle.py` verifies the tag and embeds the source binding into the packaged repository bundle. Signing, notarization, bytecode sealing, and CI invariants are documented in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md).

### macOS (.dmg)

```bash
bash scripts/download_python_standalone.sh
OUROBOROS_SIGN=0 bash build.sh
```

Output: `dist/Ouroboros-<VERSION>.dmg`, containing `Ouroboros.app` and `Install CLI.command`. Omit `OUROBOROS_SIGN=0` when a Developer ID signing identity is configured.

### Linux (.tar.gz)

```bash
bash scripts/download_python_standalone.sh
bash build_linux.sh
```

Output: `dist/Ouroboros-<VERSION>-linux-<arch>.tar.gz`, containing `Ouroboros/bin/install-ouroboros-cli`. If bundled browser tools need host libraries, run `./Ouroboros/python-standalone/bin/python3 -m playwright install-deps chromium webkit`.

### Windows (.zip)

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_python_standalone.ps1
powershell -ExecutionPolicy Bypass -File build_windows.ps1
```

Output: `dist\Ouroboros-<VERSION>-windows-x64.zip`, containing `Ouroboros\bin\install-ouroboros-cli.cmd`.


## Architecture and Runtime Data

The native launcher starts a web runtime and supervisor-managed agent workers. The agent core lives in `ouroboros/`, the interface in `web/`, the process plane in `supervisor/`, and the runtime's durable identity, state, history, logs, and skills under `~/Ouroboros/data/`.

The full component map, data flow, API surface, storage layout, safety boundary, and operational rationale live in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Deployment details live in [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

## Runtime Commands

| Command | Purpose |
|---------|---------|
| `/panic` | Stop the runtime and its managed processes immediately. |
| `/restart` | Restart without automatically resuming the active owner task. |
| `/status` | Show workers, task queue, and budget state. |
| `/evolve on\|off` | Start or stop autonomous evolution. |
| `/review` | Queue a deep constitutional and architectural self-review. |
| `/bg start\|stop\|status` | Control background consciousness. |


## Philosophy

The 13 Constitution principles — Agency, Continuity, Meta-over-Patch,
Immune Integrity, Self-Creation, LLM-First, Authenticity & Reality
Discipline, Minimalism, Becoming, Versioning and Releases, the absorbed
Iterations / Spiral lineage, and Epistemic Stability — are defined in
full in [`BIBLE.md`](BIBLE.md). That file is the constitutional SSOT
(Bible P4 Ship-of-Theseus protection) and this README intentionally does
not paraphrase it.

---

## Contributing

External contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md)
for the complete workflow. Open pull requests against the lowercase
`ouroboros` branch and leave release-version allocation to maintainers. A
current OpenRouter triad + scope packet is the optional fast path; pull
requests without one remain welcome but require more maintainer-side review
and integration work.

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| 6.74.5 | 2026-07-22 | **fix: subagents can read the skill payloads they audit; budget drift compares like with like.** (1) v6.70.0 granted read-only scouts read/list/search on `root=skill_payload`, but the path layer still resolved payloads against the child's isolated drive (`data/state/headless_tasks/<tid>/data`), which physically has no `skills/` tree — every scout was blinded with a bare "Directory not found" (2026-07-21 anime_studio audit swarm: three children produced zero payload reads and the parent died budget_exhausted doing everything alone). `resource_root_path` now resolves `skill_payload` against the canonical data root (new `canonical_data_root` helper: task_metadata `budget_drive_root` → ctx `budget_drive_root` → `drive_root`), so root tasks and isolated benchmark roots are unchanged while children read the real payload; the verb matrix is untouched — write/edit/review stay parent-only, path confinement and control-plane sidecar guards unchanged, native bucket stays out of the data-plane resolver. (2) `budget_drift_alert` compared the ALL-provider ledger delta against ONE OpenRouter key's usage delta, so real direct-provider spend (Anthropic advisory ~$98/day) latched the alert at ~88% while nothing was wrong. Drift now compares the OpenRouter-only settled ledger delta (`by_provider` from the attempt ledger; settled-only, reservations excluded) against the key's usage delta, rebaselines silently when the configured key changes mid-session (non-secret sha256 fingerprint) or when a pre-upgrade state lacks the new snapshot, suppresses the comparison honestly while the ledger is integrity-degraded, and `status_text` renders exactly the same deltas as the computation (the warning event keeps the all-provider delta as context). |
| 6.74.4 | 2026-07-21 | **feat: workspace-tree freeze directives (mitigation) + truthful ProgramBench submission contract.** Root cause (PB cmatsuoka__figlet smoke): an agent committed a compiling state, then broke the tree with one last uncommitted edit as the acceptance improvement loop hit its pass cap — and the harness ships the LIVE tree (`.git` dropped), so the verified commit protected nothing. All existing salvage machinery guards the answer TEXT only. Fix, prompt-only (P5) over existing channels (P7): the acceptance rails line marks the last admitted improvement pass (`passes_done+1 >= cap`, within cap>0) as FINAL, and EVERY workspace-delivery capsule (canonical `is_workspace_mode()` authority, attribute fallback for light contexts) carries the tree directive — a deadline or cost rail can end the loop between capsules — keep the tree VERIFIED (rebuild, verify, and commit if the task calls for a commit; revert unverified edits); the 10% deadline flush AND the ~80% cost wrap-up gain one shared commit-NEUTRAL tree sentence (acting self_worktree subagents cannot commit; a moved HEAD fails patch capture closed), byte-identical for non-workspace tasks; the ProgramBench instruction now states the true submission model — a source tarball from the CURRENT tree state (uncommitted edits DO ship; `.git`, root binaries and build/cache noise excluded), run `./compile.sh` one final time — replacing the false "fresh checkout" framing. Disclosed residual (mitigation, not closure): a forced tool-less exit — deadline grace or budget stop crossed inside one long round, with no pacing note or capsule in the terminal stretch — can still ship an unverified last edit; the structural verification-freshness seam is a filed follow-up pending an owner decision. |
| 6.74.3 | 2026-07-21 | **fix: Windows portability of one v6.74.0 guard test.** `test_genuine_repo_target_still_blocks` built its shell command via an f-string embedding a Windows path (backslashes mangled by shlex) and failed the 3-OS full matrix on windows-latest; the test now passes argv lists. No runtime code changes. |
| 6.74.2 | 2026-07-21 | **fix: CI portability of the two new GAIA sandbox-staging tests.** They imported `inspect_ai` directly — an optional benchmark dependency absent on CI runners — and failed quick-test with ModuleNotFoundError. The tests now inject a fake `inspect_ai.util.sandbox` module via monkeypatch, keeping the success-path coverage on every environment. No runtime code changes. |
| 6.74.1 | 2026-07-21 | **fix: CI lint gate — remove one unused test import.** The v6.74.0 tag CI failed on the deterministic ruff F-rule gate: `tests/test_devtools_benchmarks.py` carried an unused `types.SimpleNamespace` import added with the final GAIA staging tests. Import removed; no runtime code changes. Fix-forward release so the v6.74.x artifacts build (the published v6.74.0 tag is never re-tagged). |
| 6.74.0 | 2026-07-21 | **feat: the acceptance review becomes a reviewer-authored terminating dialogue.** The improvement capsule now LEADS with the actual outcome — aggregate verdict + tier + the real blocker (one shared `panel_reason` reducer feeds capsule, projection, and progress lines) — plus the concrete open obligation ids and one rails line naming every active termination source with its remaining headroom (money/time/rounds/review passes, each from its real source). The do-nothing tail is replaced by the three real moves: FIX the work, REBUT structurally via `obligation_dispositions`, or DECLARE a requirement unreachable here; the obligations clause records disagreement ONLY via dispositions. Obligation identity becomes reviewer-authored: findings carry `disposition_kind` (`new`/`re_raise`); a `re_raise` MUST name an existing catalog id (unknown ids fail closed to `new`, disclosed), and a re-raise REOPENS the row without wiping the agent's argument (`previous_disposition`/`previous_reason`/`reopened_count` survive; the reviewer sees the prior argument and adjudicates it with the commit gate's rebuttal contract). Each acceptance reviewer also emits a typed `dialogue_status` (`continue_actionable`/`unreachable_here`/`stable_disagreement`); a NEW pure reducer over ALL contract-valid actors (not the aggregate-filtered set) applies the panel's own quorum — any contributing continue keeps the loop; a quorum of terminal votes finalizes through the existing honest `best_effort_open_obligations` path with both positions recorded. One short reachability clause + review-register framing joins the acceptance prompt (an outside perspective, not a gate; unreachable-here requirements are classified honestly, never re-raised as blocking). Reviewer prompts split into two cache-marked segments (byte-stable governance + task-stable contract) with the mutable evidence tail unmarked and the slot label moved off byte 0 (concurrent same-model slots share a warm prefix; ≤4 breakpoints asserted on the final payload). Harness truthfulness: GAIA stages official sandbox `Sample.files` via `sandbox().read_file` with exact-relative shared-root lookup + per-file provenance (a declared-but-unresolvable attachment is a typed infra error); a tracked CLB operator patch populates `acceptance_claims` in all three task-body writers + the knowledge-topic steer nudge + a bounded cost-finality wait; the SWE-Pro shard budget is derived as `per_task_cost × scheduled_tasks` (total==cap starvation fails loudly, and `run_pro.py` seeds each task's `TOTAL_BUDGET` from the cumulative ledger spend on the first task of an invocation too — the `i > 1` fast-path defeated the derived total on per-task auto_run drives); CLI/PB readers wait (bounded) for `task_cost_finalized` only on `completed`/`degraded`. Runtime fixes: the light-mode shell guard resolves cwd BEFORE judging repo targets (a resource-root label cwd no longer false-blocks task-drive writes; resolution failure fails closed), and the post-task cost publish uses `try_get_bridge` so headless finalization stops warning about an uninitialized message bus. Governance: the commit/plan minimalism items gain the generative surface-duty (name the existing ARCHITECTURE mechanism when a diff/plan adds a surface). |
| 6.73.0 | 2026-07-20 | **feat: project origin invariant — the start-message-loss class is closed.** The identity of the owner message that starts a project is now CAPTURED AT CHAT INGRESS (where the canonical `chat.jsonl` row is written) and passed BY VALUE through every path — `promote_chat_to_task`, `route_to_project`, `ensure_project_scope`, direct project-room turns, and post-hoc UI conversion (which reads it from the persisted task record). `bind_task_to_project` requires a TYPED origin: the ingress-captured ref (+the full `source_text` for cross-thread origins — the retention-proof copy) or a closed-enum absence reason; omission raises, and a same-project re-bind may one-way enrich a ref-less row (a valid ref is never changed; bindings gain `_schema_version` 1). The content-hash identity lookup (`find_owner_message_ref`) — the root cause behind four serial start-message-loss fixes — is DELETED, and a new DEVELOPMENT.md anti-pattern (content-derived identity for host-minted records) pins the class. The Project history lens synthesizes the start message from the binding's own text when the canonical row left the bounded read window (post-quota, identity-deduped, hard-capped, `origin_projected=true`), so an old project's origin survives any number of chat-log rotations. Same-class neighbors fixed: the memory consolidator's cursor is generation-aware (locates its stored `chat_log_signature` across the ordered archive chain and consolidates `archives[i:]+live`, per-segment signature discipline — up to 99 messages were silently dropped at EVERY ~800KB rotation; an unfindable generation now appends an explicit durable `[MEMORY GAP]` block), a failed durable bind is loud (`log.warning` + typed `project_binding_failed` event) instead of `except: log.debug`, and verification-receipt (`except: pass`) / reflection memory-action (`except: log.debug`) write failures now warn instead of vanishing. |
| 6.72.0 | 2026-07-18 | **feat: chronological chat history and honest UI presentation.** Chat bubbles, live-card roots, photos, videos, and documents now carry sortable epoch `data-ts` values from their raw source timestamps. Timeline insertion orders against the first strictly newer sibling, preserves equal-timestamp arrival order, keeps typing last, and retains append behavior for timestamp-free nodes; inserting older history above a scrolled-up viewport compensates `scrollTop`, while near-bottom autoscroll stays unchanged. Two-pass replay keeps progress-only, terminal-without-summary, disconnected, and nested-subagent cards intact, with task cards anchored by their earliest source event and non-today live/log timestamps showing a date. `FINAL ANSWER:` remains the unchanged protocol-gated backend latch/extractor contract but renders as ordinary assistant/system text instead of an Answer capsule across live, history, and reconnect paths. The chat composer’s Context Mode copy now describes the hot-applied owner setting without promising a next-task boundary, the unchanged busy-state 409 names queued as well as running work, and the chat header reserves top padding from its real measured height so wrapped narrow-viewport headers no longer clip the first message. |
| 6.71.0 | 2026-07-17 | **feat: UI polish — a shared rich-content contract, one composer dock, calm charts, and whole-row disclosures.** Rendered markdown and review disclosures adopt one `.ui-rich-content` contract (reserved list gutter, anywhere-wrapping, no marker bleed outside the card) applied to widget markdown and the Skills review history/findings; the skill-review bubble gets a symmetric bottom timestamp inset. Live-line disclosures toggle from the WHOLE non-interactive row surface (nested buttons/links/selection keep their own behavior; focus lands on the real toggle button; Enter/Space/aria-expanded unchanged) with a larger expand label. Project/side-chat composers dock exactly like the main chat (same absolute overlay, same bottom fade, same scroll reserve — no second fade layer). Widget charts render in a bounded 260-360px box and poll ticks update `chart.data` in place instead of destroy/recreate flicker; background refetches keep the content and show a thin pulsing indicator instead of a loading swap (loading only when there is nothing to show yet). New node tests pin the disclosure guards; the declarative Playwright smoke gains a markdown fixture with geometry asserts (list markers inside the card box) and a bounded-canvas assert. |
Older releases are preserved in Git tags and GitHub releases. Older 6.x rows (including 6.73.2, 6.73.1, 6.70.0, 6.71.1, 6.71.2, 6.65.4, 6.69.0, 6.65.3, 6.65.2, 6.68.0, 6.67.0, 6.65.1, 6.66.0, 6.65.0, 6.64.3, 6.64.0, 6.64.2, 6.64.1, 6.63.0, 6.62.0, 6.61.3, 6.61.4, 6.61.0, 6.61.1, 6.60.0, 6.59.0, 6.54.4, 6.58.0, 6.57.0, 6.56.0, 6.55.0, 6.54.2, 6.54.1, 6.54.0, 6.53.4, 6.53.0 and 6.51.0), the 5.2.0 through 5.33.0-rc.6 rows, and former `4.0.0` rows are rolled off to respect the P9 changelog cap; their full bodies remain at their git tags.

---

## License

[MIT License](LICENSE)

Created by [Anton Razzhigaev](https://t.me/abstractDL) & Andrew Kaznacheev
