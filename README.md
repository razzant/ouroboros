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
[![Version 6.86.1](https://img.shields.io/badge/version-6.86.1-green.svg)](VERSION)

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
| 6.86.1 | 2026-07-31 | **feat: Git commits credit Ouroboros by default without taking attribution out of the dialogue.** The system prompt asks Ouroboros to add the GitHub-recognized `Co-authored-by: Ouroboros <311266734+ouroboros-agent@users.noreply.github.com>` trailer when it commits work it contributed to. This remains LLM-first and conversational: the human can omit, pause, resume, or scope attribution through ordinary dialogue and memory, with no Git hook, keyword classifier, settings flag, or commit-tool rewrite. Ouroboros avoids a duplicate trailer when it is already the primary author and preserves existing author and co-author credits. |
| 6.86.0 | 2026-07-30 | **feat: the OSWorld working prompt gains an atomic task contract, and a task's proxy session stops leaking into the published tree.** Forensics on the v6.84.0 run (89.05%) against the leaderboard leader's own published per-task dump showed the gap is 19 tasks, 8 of them one class: the work was done and never checked against the surface the grader reads. The agent now writes the task's obligations as a numbered checklist BEFORE its first mutating action — object, required state with every stated qualifier, order, what must stay unchanged, where the result must persist — and closes each item as observed-satisfied / not-verified / impossible before it may finish, with at most one targeted repair. Plural instructions still cover every matching element; only a SINGULAR referent resolving to several candidates forces a justified choice of one, and the contract is explicitly revisable when observation contradicts it. Three infeasibility shapes are named: discovery that falls outside a stated means restriction, a named mode of operation the application does not ship, and a mechanism whose trigger is narrower than the task states. The desktop environment's own configuration CLI (gsettings/dconf) is declared a legitimate surface — it writes the same store the Settings app does — while private application state stays forbidden. The colour clause drops a motivation that was simply untrue on a graded task (the metric there measures distance from a pure primary and reads no reference file at all). Harness: each proxy-flagged task now gets its own sticky upstream session keyed on campaign root plus example id, so two concurrent campaigns never share an exit; that config carries the account password and is therefore written to LANE-PRIVATE state and unlinked afterwards, never under `results/`, which is the tree that gets archived and published — an earlier draft of this same change wrote it into all 361 result directories. After the post-gate reset the runner also probes whether the binaries a task's setup claims to install are actually present and records the answer in the manifest, because upstream reports a guest command that failed as "executed successfully", and a premise that vanished between gate and worker otherwise surfaces as an honest infeasible scored zero. |
| 6.85.0 | 2026-07-30 | **feat: Telegram becomes a first-party native capability, and blocked skill repairs regain a valid completion path.** (1) The bundled `telegram` skill consolidates the proven owner-only text/photo bridge and Mini App PoC without migrating, disabling, deleting, or changing either legacy payload. It preserves the existing bridge commands, outbound media, cards, opt-in notifications, mirror-all behavior, Ouroboros SPA, private first-contact binding, process-memory sessions, pinned Quick Tunnel lifecycle, menu rollback, and platform limits; the Mini App may be disabled or unavailable while the text bridge remains loaded. Native trust is hash-bound, while the bot token and privileged host permissions still require the normal Grant access then enable flow. (2) Bounded manifest `conflicts` declarations are enforced symmetrically at enable, reconcile, startup, and dispatch, returning a typed conflict without automatic state transfer. The Skills card now says `Loaded` for extension registration instead of overstating readiness as `Active`, and Telegram reports bridge and Mini App status in its own surface. (3) Typed `skill_repair` requests are promoted to managed tasks before ephemeral routing, preserving payload confinement, review access, and `allow_enable=false`; ordinary ephemeral default-deny policy is unchanged. (4) Google Colab discovers the seeded native Telegram skill, waits for a fresh executable native verdict, grants only API-reported missing grantable items under the persisted owner policy, enables it, and saves the proven full-access, mirror, and Mini App defaults without a Hub install or extra review. |
| 6.84.0 | 2026-07-30 | **fix: the OSWorld working prompt stops charging the wrong resource, and three of its own clauses stop costing points.** Forensics over every failed task of the v6.83.0 dual run (74 agents, whole-loss coverage: the per-task deltas sum to the measured 12.90 pp) found the most expensive defect was ours: the preamble said "every tool call costs ~30s, so MINIMIZE calls" while the budget is denominated in TURNS and the official contract batches actions inside one `predict()`. The agent obeyed — 1.01 tool calls per turn across 11k turns, i.e. the benchmark ran on about a third of its action budget. The clause is now turn-denominated and asks for 4-8 confident calls in one turn, split only where a target depends on what the previous action reveals. Three clauses the agent CITED while losing are corrected: an exact value now beats the app's named swatch (a palette "Blue" 2A6099 is not 0000FF); "already in the requested state" must be judged from the STORED value, since controls render defaults as selected while nothing is stored; and ordinals no longer blanket-exclude headings, which on a slide are often the counted item. The command line is restored as the right tool for genuine batch/file work (one `pdfseparate` instead of N print dialogs) while GUI stays mandatory for application state. Added: verification by independent read-back (re-open the saved artifact, read it with a different tool than wrote it) merged with the minimal-diff rule into one clause; and a premise branch — a task asking for something VISIBLE is not satisfied by storing a flag, and an agent writing that the real path is impossible or that it is delivering a stand-in has already found its verdict. Adapter/prompt only. Harness: `evaluate()` now runs with the checkout root as CWD, because evaluator fixtures are declared relative and `get_local_file` tests them against the process CWD — one task produced a byte-exact answer and scored 0; the gate's UNUSED turn reserve is returned to the worker (the gate budgets 14 and spends ~4, and 13 of 56 failures died at 89-92 turns inside a 100-turn budget); and proxy support is gated on a LIVE CONNECT probe rather than the config file existing, with a proxy:true task whose trace shows an exhausted upstream quarantined as infrastructure instead of scored as a capability zero. |
| 6.83.0 | 2026-07-30 | **fix: a screenshot that cannot be decoded fails where it is taken, an infeasibility verdict is judged as an argument, and a declared step budget is one the runtime actually enforces.** (1) Image integrity is fail-closed at three seams: the remote screenshot fetch validates a FULL decode before publishing a path (bounded re-fetch, write-validate-rename), the shared remote-result builder rejects an undecodable capture instead of claiming ok, and the VLM payload builder raises `IMAGE_UNDECODABLE` at build time. A truncated PNG keeps a valid 24-byte header, so header-only checks passed it and it detonated rounds later as a non-retryable provider 400 — five task deaths in the v6.81.1 OSWorld run. Four test fixtures labelled "minimal valid PNG" were themselves undecodable and are now real images; one assertion that pinned an IDENTITY coordinate transform for a 1920x1080 capture at a 1280 cap (it only held because the stub never downscaled) now pins the real 1.5x transform. (2) Tool results are judged by their typed envelope: a structured `{"ok": false}` payload counts as a failure for the error counters, anti-loop and auto-attach, instead of only text markers. (3) Acceptance review gains an ABSENT-PREMISE branch: when the terminal claim is that the premise is missing, the deliverable under review is the PREMISE ARGUMENT — instantiating "the named artifact exists" as a criterion begs the question, and coaching a continuation whose remaining routes breach the task's own stated restrictions manufactures the artifact the task forbids. A weak premise argument still fails on its own grounds. Measured cost of the old behaviour: a correct 1.0 converted into 0.0 over 149 tool calls. (4) `type_text` routes multi-line and long payloads through the in-VM clipboard (typewrite presses Enter per newline and sheds keystrokes), joining the non-ASCII and angle-bracket paths. (5) OSWorld adapter: `--max-steps` declares AND enforces a leaderboard-comparable budget — a step is one top-level policy turn, matching the official `predict() -> actions[]` boundary, not one GUI action; the server round cap is verified against the derived worker cap before the VM boots, the gate phase is cancelled at its own reserve (the runtime cap is server-wide and the gate is a separate task), and the post-run audit reads the loop's policy turns rather than the flat physical-call field, which disagreed with it on 344 of 346 examples. `--expect-dataset-commit` turns the graded-spec pin into a gate: a checkout other than the campaign's is refused before any paid work, because it supplies different task instructions AND a different evaluator. |
| 6.82.0 | 2026-07-29 | **feat: the Working card tells the truth again, rarely-used providers step aside, mobile gets two real gestures, the shipped model set moves forward, and a run can be cancelled where it is shown.** (1) A collapsed live card carries a dedicated activity line beside its identity: the title keeps the coined name (or a child's role-model-id), the line shows the latest meaningful action (bounded at 400 chars with a visible ellipsis, complete text on the element), an unnamed card does not duplicate itself, and a finished card keeps its last activity — the regression introduced when proactive naming took the single title slot in v6.40.0. (2) Card cost is honest and sticky: only frames with task-scope accounting evidence may set it (a per-round `llm_round_finished` delta is NOT task cost), rank is unavailable < pending < final, and a costless frame re-renders the stored value instead of erasing it. Reload replays the same truth: the snapshot's flat cost fields ride on `task_summary` rows (up to nine; `cost_usd` and `cost_accounting_error` come with the persisted terminal result, which overrides the row). (3) Cloud.ru and the OpenAI-compatible pair move into a collapsed onboarding "More options" group, Cloud.ru and GigaChat into a Settings "More providers" section that auto-opens for a usable credential; every input stays mounted, so load/save is unchanged. The About footer drops its lab byline. (4) Two mobile gestures and no more: swipe-left closes the drawer, swipe-down on the project-panel header closes the overlay — release-triggered through one small `gestures.js` (pure classifier + binder, selection/editable/scroller guards, scoped click suppression, disabled with the keyboard open), no drag-follow, no other surfaces. (5) Shipped defaults move to grok-4.5 main, gemini-3.6-flash light, gpt-5.6-luna fallback, gpt-5.6-sol-pro deep review, a gpt-5.6-terra + gemini-3.6-flash + deepseek-v4-pro triad and a gpt-5.6-terra scope reviewer (1M window; the sentinel and both migration sets moved with it, so an upgraded install cannot keep a retired sub-floor reviewer). Installs whose settings file never stored a model key follow the new defaults by design. Stored values equal to a FORMER SHIPPED DEFAULT migrate forward only where the provider-defaults path already runs — exclusive direct-provider installs and the scope-review slot; an ordinary OpenRouter install keeps its stored models (the owner's file is not rewritten), and a value that differs from every shipped default is never touched anywhere. Disclosure: an Anthropic-only direct install's auto-filled review triad is now the loud single-model [sonnet-5]×3 (main==light), a deliberate diversity trade-off; the deep-self-review slot now ships a per-provider value too (direct OpenAI gets plain `gpt-5.6-sol` — the shipped OpenRouter default's `-pro` is a router slug that 404s on api.openai.com, and pro reasoning lives only on the Responses API this codebase does not call, so a direct-OpenAI deep review is deliberately the non-pro model; anthropic gets opus-5; the prior shipped value migrates), so a direct-only install no longer keeps an unreachable OpenRouter-form id for it — Cloud.ru and GigaChat get NO auto-filled deep slot, because they are documented below the 1M window deep review sizes against. Safety `light` is authored only by a FRESH desktop wizard through a narrow flag proved under the settings lock — the shipped default and every fail-closed fallback stay `full`, and existing, web and Docker installs keep `full`. (6) The per-root active-subagent ceiling rises 50 to 500 through one shared constant (default stays 6; `wait_tasks` shares it, and the accepted scale trade-offs are documented). (7) A live root card gains **Cancel run**: it cancels the task and its live subtree (re-sweeping late-admitted descendants) and answers only once the teardown finished — one synchronous transaction off the event loop, so the split-transaction family (pre-ack latch, partial-record statuses, background fence ownership) is absent rather than guarded. Cancellation takes CUSTODY (capture under the lock marks the worker slot, kill/join run outside it and must CONFIRM death, publication happens only after that; any failure restores custody and refuses), and outcomes are typed per task, so a child that refused to die fails the whole cascade with a 503 instead of a false success; a natural-completion race is a graceful no-op and a settled task keeps its own outcome, and renders a cancelled task as "Cancelled" instead of a generic "Done" on every surface. Text steering stays cooperative; Panic remains the only global stop. (8) Parsed git output is locale-pinned, so a non-English git no longer breaks commit-binding checks. |
| 6.81.1 | 2026-07-29 | **fix: one round per look, one verdict per premise — the OSWorld v6.81.0 full-run forensics land as mechanism fixes.** (1) Tool results may now carry a typed `auto_attach_image` capability: the host attaches that local image to the conversation in the SAME round, through the exact implementation the `view_image` tool uses (one shared body — identical allowed-roots/size/MIME perimeter, identical durable copy under `uploads/views`, identical message shape), strictly non-fatally. The unix_computer_use screenshot emits it; the measured cost it removes: 3,830 of 16,367 rounds in the v6.81.0 OSWorld run (~21% of the whole round budget) were the mandatory second `view_image` round per observation, and every task that reached the 200-round cap scored 0. `MAX_LIVE_IMAGE_BLOCKS` 3→5 (owner decision) so auto-attached screenshots keep some visual history. (2) unix_computer_use: one pointer-coordinate normalizer accepts the malformations models actually emit (the pair packed into `x` with `y` absent or duplicated — 109 wasted rounds in one run), y leaves the required-schema so recovery happens before binding, ambiguity still fails loudly; `double_click`/`triple_click` register as thin click aliases (111 previously-"Unknown tool" calls); `remote_exec`'s description now states its real contract (fresh `bash -lc` in $HOME — not the visible terminal, no cwd/history inheritance). (3) OSWorld gate v3: the premise prompt becomes a structured rubric (action → referent → blocking → acquirable → store-or-render → unbound placeholders) replacing the example list whose false kills all judged outcome-meaningfulness instead of action-performability; the confirming challenger is REMOVED on its own full-run ledger (20 invocations, 0 saves, 1 officially-infeasible task lost, 215 worker rounds burned, and it confirmed all four false kills — identical-prompt re-reads are correlated, not independent); the working preamble names the graded traps (live-window re-save, visible-terminal history, pkill self-match). Adapter-only where possible; the three core touches are the attach hook in `process_tool_results`, the `view_image` body factored into the shared `vision.attach_local_image_to_context` seam, and the image-budget constant. |
| 6.74.5 | 2026-07-22 | **fix: subagents can read the skill payloads they audit; budget drift compares like with like.** (1) v6.70.0 granted read-only scouts read/list/search on `root=skill_payload`, but the path layer still resolved payloads against the child's isolated drive (`data/state/headless_tasks/<tid>/data`), which physically has no `skills/` tree — every scout was blinded with a bare "Directory not found" (2026-07-21 anime_studio audit swarm: three children produced zero payload reads and the parent died budget_exhausted doing everything alone). `resource_root_path` now resolves `skill_payload` against the canonical data root (new `canonical_data_root` helper: task_metadata `budget_drive_root` → ctx `budget_drive_root` → `drive_root`), so root tasks and isolated benchmark roots are unchanged while children read the real payload; the verb matrix is untouched — write/edit/review stay parent-only, path confinement and control-plane sidecar guards unchanged, native bucket stays out of the data-plane resolver. (2) `budget_drift_alert` compared the ALL-provider ledger delta against ONE OpenRouter key's usage delta, so real direct-provider spend (Anthropic advisory ~$98/day) latched the alert at ~88% while nothing was wrong. Drift now compares the OpenRouter-only settled ledger delta (`by_provider` from the attempt ledger; settled-only, reservations excluded) against the key's usage delta, rebaselines silently when the configured key changes mid-session (non-secret sha256 fingerprint) or when a pre-upgrade state lacks the new snapshot, suppresses the comparison honestly while the ledger is integrity-degraded, and `status_text` renders exactly the same deltas as the computation (the warning event keeps the all-provider delta as context). |
| 6.74.4 | 2026-07-21 | **feat: workspace-tree freeze directives (mitigation) + truthful ProgramBench submission contract.** Root cause (PB cmatsuoka__figlet smoke): an agent committed a compiling state, then broke the tree with one last uncommitted edit as the acceptance improvement loop hit its pass cap — and the harness ships the LIVE tree (`.git` dropped), so the verified commit protected nothing. All existing salvage machinery guards the answer TEXT only. Fix, prompt-only (P5) over existing channels (P7): the acceptance rails line marks the last admitted improvement pass (`passes_done+1 >= cap`, within cap>0) as FINAL, and EVERY workspace-delivery capsule (canonical `is_workspace_mode()` authority, attribute fallback for light contexts) carries the tree directive — a deadline or cost rail can end the loop between capsules — keep the tree VERIFIED (rebuild, verify, and commit if the task calls for a commit; revert unverified edits); the 10% deadline flush AND the ~80% cost wrap-up gain one shared commit-NEUTRAL tree sentence (acting self_worktree subagents cannot commit; a moved HEAD fails patch capture closed), byte-identical for non-workspace tasks; the ProgramBench instruction now states the true submission model — a source tarball from the CURRENT tree state (uncommitted edits DO ship; `.git`, root binaries and build/cache noise excluded), run `./compile.sh` one final time — replacing the false "fresh checkout" framing. Disclosed residual (mitigation, not closure): a forced tool-less exit — deadline grace or budget stop crossed inside one long round, with no pacing note or capsule in the terminal stretch — can still ship an unverified last edit; the structural verification-freshness seam is a filed follow-up pending an owner decision. |
| 6.74.3 | 2026-07-21 | **fix: Windows portability of one v6.74.0 guard test.** `test_genuine_repo_target_still_blocks` built its shell command via an f-string embedding a Windows path (backslashes mangled by shlex) and failed the 3-OS full matrix on windows-latest; the test now passes argv lists. No runtime code changes. |
Older releases are preserved in Git tags and GitHub releases. Older 6.x rows (including 6.76.0, 6.75.0, 6.74.1, 6.74.0, 6.73.2, 6.73.1, 6.73.0, 6.72.0, 6.71.2, 6.71.1, 6.71.0, 6.70.0, 6.69.0, 6.68.0, 6.67.0, 6.66.0, 6.65.4, 6.65.3, 6.65.2, 6.65.1, 6.65.0, 6.64.3, 6.64.2, 6.64.1, 6.64.0, 6.63.0, 6.62.0, 6.61.4, 6.61.3, 6.61.1, 6.61.0, 6.60.0, 6.59.0, 6.58.0, 6.57.0, 6.56.0, 6.55.0, 6.54.4, 6.54.2, 6.54.1, 6.54.0, 6.53.4, 6.53.0, 6.51.0), the 5.2.0 through 5.33.0-rc.6 rows, and former `4.0.0` rows are rolled off to respect the P9 changelog cap; their full bodies remain at their git tags.

---

## License

[MIT License](LICENSE)

Created by [Anton Razzhigaev](https://t.me/abstractDL) & Andrew Kaznacheev
