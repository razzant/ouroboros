# Ouroboros

<a href="https://github.com/oslook/github-trending/blob/1d61d20a46f66a9590286bf23a8ce8db99be3acf/2026-08-04/python_weekly_trending.json"><img src="assets/github-trending.svg" width="250" height="55" alt="GitHub Trending: #9 Python weekly, August 2026"></a>

[![GitHub stars](https://img.shields.io/github/stars/razzant/ouroboros?style=flat&logo=github)](https://github.com/razzant/ouroboros/stargazers)
[![Downloads](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Frazzant%2Fouroboros%2Fbadges%2Fdownloads.json)](https://ouroboros-agent.ai/install/)
[![Website](https://img.shields.io/badge/website-ouroboros--agent.ai-c93545.svg)](https://ouroboros-agent.ai/)
[![Technical report](https://img.shields.io/badge/technical_report-arXiv%3A2608.08311-b31b1b.svg)](https://arxiv.org/abs/2608.08311)
[![Best of Agent Harnesses](https://img.shields.io/badge/%F0%9F%8F%86_Best_of-Agent_Harnesses-5ac4bf)](https://github.com/RyanAlberts/best-of-Agent-Harnesses#ouroboros-2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![macOS 12+](https://img.shields.io/badge/macOS-12%2B-black.svg)][download-macos-arm64]
[![Linux](https://img.shields.io/badge/Linux-x86__64-orange.svg)](https://ouroboros-agent.ai/install/#linux)
[![Windows](https://img.shields.io/badge/Windows-x64-blue.svg)][download-windows-x64]
[![OuroborosHub](https://img.shields.io/badge/OuroborosHub-skills%20marketplace-8A2BE2.svg)](https://github.com/razzant/OuroborosHub)
[![Version 7.9.0](https://img.shields.io/badge/version-7.9.0-green.svg)](VERSION)

Ouroboros is an open-source, general-purpose AI agent whose identity, durable memory, and history continue across tasks and restarts. It works on external projects, coordinates a live swarm of specialist agents, and can rewrite the implementation it runs on, including its code, architecture, prompts, tools, and dependencies. Reflection can also change how it understands itself without severing that continuity.

It runs as a native desktop app or through a headless CLI. The runtime keeps its repository, durable memory, history, and interface on your machine, while model inference can use remote APIs you configure or a local GGUF model.

Ouroboros sends no telemetry by default, and the official project operates no usage or crash-report collection. A deployment can connect its own monitoring tools to its local records and choose what they receive; local history stays on the machine either way. Requests through OpenRouter carry the app name, which OpenRouter uses for its public app statistics.

> **Changing Ouroboros? Coding agents and people must read [CONTRIBUTING.md](CONTRIBUTING.md) before editing.** It defines the required project context, verification, and separate-agent review flow.

## Download Ouroboros

> **Just want to use Ouroboros? Click the download for your platform below. You do not need to clone this repository or install Python or uv.**

Desktop downloads bundle CPython 3.10.22 and SQLite 3.53.1. In-app Update replaces the repository and its packages, not that interpreter: install the app again to get this runtime.

- **macOS 12+ on Apple silicon:** [**Download for macOS (.dmg)**][download-macos-arm64]
- **Windows x64:** [**Download for Windows (.zip)**][download-windows-x64]
- **Debian, Ubuntu, or Astra Linux x86_64:** [**Download the Debian package (.deb)**][download-linux-deb-amd64]
- **Fedora or RHEL x86_64:** [**Download the RPM package (.rpm)**][download-linux-rpm-x86_64]
- **RED OS 8 x86_64:** [**Download the RED OS package (.rpm)**][download-linux-rpm-red80-x86_64]
- **Other Linux x86_64:** [**Download the portable AppImage**][download-linux-appimage-x86_64] or the [tar.gz archive][download-linux-x86_64]

**Android (experimental, Magisk-rooted ARM64):** [Installation and qualification guide](docs/ANDROID_INSTALL.md). Its USB setup requires Python and adb.

Files named `SHA256SUMS`, `release-evidence.json`, `release-smoke-*.json`, and `sbom-*.cdx.json` are verification evidence, not additional installers.

### macOS quick start

1. Click [**Download for macOS (.dmg)**][download-macos-arm64]. The current file is named `Ouroboros-<version>.dmg`.
2. Open the DMG and drag `Ouroboros.app` onto the **Applications** shortcut.
3. Open Ouroboros from Applications. If macOS refuses to open it, choose **Open Anyway** in **System Settings → Privacy & Security** (on macOS 14 and earlier, right-clicking the app and choosing **Open** also works).

<p align="center">
  <img src="assets/install-macos.png" width="760" alt="Ouroboros DMG window with a large arrow from Ouroboros.app to the Applications shortcut and Install CLI.command below">
</p>

### Homebrew on macOS

Homebrew can install and update the same notarized app. This repository is its own tap:

```bash
brew tap razzant/ouroboros https://github.com/razzant/ouroboros.git
brew install --cask razzant/ouroboros/ouroboros
```

- Each Stable promotion updates the cask. Homebrew upgrades only when the cask's version differs from the version it recorded when it installed or adopted the app; then a plain `brew upgrade` compares the installed app version on current Homebrew (older versions skip `auto_updates` casks), and a named `brew upgrade --cask ouroboros` or `--greedy` can replace even a newer self-updated app with the cask's Stable version, including its bundled runtime. `brew reinstall` installs the cask's version over whatever app is there.
- An upgrade replaces the app and its bundled runtime but keeps the managed source checkout (`~/Ouroboros/repo` by default), so Ouroboros goes on running the source it had, for example 7.7.1 inside the 7.8.0 app, until you run Update in the app. In the released 7.8 app, in-app Update changes source, not the app bundle.
- If Ouroboros.app is already in Applications, `--adopt` takes it over without comparing versions and records the cask's version, so an older adopted app stays as it is and `brew upgrade` reports it up to date until the next Stable promotion; `brew reinstall --cask ouroboros` or `--force` replaces it.
- Homebrew asks a running Ouroboros to quit before an upgrade or uninstall and reopens it after an upgrade, without your shell environment. If you start Ouroboros with root variables for that launch only, such as `OUROBOROS_APP_ROOT` or `OUROBOROS_DATA_DIR`, quit it first and start it again the same way.
- The first cask installs 7.8.0. Finish Setup before the first Quit, including a quit by Homebrew: otherwise this version can leave a new data folder that later starts refuse to initialize (#2412). If affected, back up the data folder (`~/Ouroboros/data` by default) and look for `prior_history_without_state` in its `logs/events.jsonl` before changing anything. Use Reset runtime data in Settings only for a confirmed empty new installation. The cask displays this warning because Stable's README remains the release's original.
- `brew uninstall --cask ouroboros`, with or without `--zap`, removes only the app: `~/Ouroboros` (repository, memory, projects, Deliverables) and any custom roots stay, and so does sign-in startup, so turn off "Start Ouroboros on the host computer when you sign in" first if you use it.
- The first `brew tap` clones this whole repository.

### Windows quick start

1. Click [**Download for Windows (.zip)**][download-windows-x64].
2. Extract the ZIP to a permanent folder.
3. Open the extracted `Ouroboros` folder and run `Ouroboros.exe`.

### Linux quick start

- On Debian, Ubuntu, or Astra Linux, download the `.deb` above and run `sudo apt install ./ouroboros_*_amd64.deb`.
- On Fedora or RHEL, download the generic `.rpm` above and run `sudo dnf install ./ouroboros-*.x86_64.rpm`. RED OS 8 has its own `red80` package.
- On another x86_64 distribution, download the AppImage, make it executable with `chmod +x Ouroboros-*.AppImage`, and run it. Git must already be installed.

To run tasks, choose a connected Codex account as the Main model source, configure a supported remote provider API key, or use a local GGUF model. Claude Code and Cursor subscriptions can run delegated coding and review; they are not Main model sources. The first-run wizard guides model access, review policy, and budget setup.

<details>
<summary>Optional CLI included with desktop downloads</summary>

The desktop packages already contain an optional CLI installer. On macOS, after copying the app to Applications, double-click `Install CLI.command` in the mounted DMG; after a Homebrew install, run `/Applications/Ouroboros.app/Contents/Resources/bin/install-ouroboros-cli`. On Linux use `./Ouroboros/bin/install-ouroboros-cli`; on Windows use `Ouroboros\bin\install-ouroboros-cli.cmd`. These installers create a user-local `ouroboros` command without sudo. You do not need Python or uv.

</details>

### Reinstall, move or remove

`~/Ouroboros` (data, memory and the code Ouroboros runs) is the installation; the app is a replaceable host. To replace or move the app, quit Ouroboros, put the new copy in place and open it: your data and code stay, and skill packages and sign-in startup follow the new copy at its next start. A new app changes only the bundled Python, SQLite and Node, never the code in `~/Ouroboros/repo`; in-app Update changes the code. **Settings → About** and the Updates page show both versions, and Ouroboros sees them in its host facts. Quit leaves the Claudexor engine running, by design.

- **macOS:** drag the new `Ouroboros.app` onto Applications and choose Replace, then open it from Applications in Finder: Spotlight can take a while to list a new copy while it indexes. Do not run Ouroboros from the disk image.
- **Windows:** keep the extracted folder in one permanent place and run it from there, never from inside the ZIP. To update, quit, remove the old folder and extract the new ZIP to the same place.
- **AppImage:** keep one file at a fixed path and replace it under the same name. **tar.gz:** keep one fixed folder. **deb/rpm:** update with your package manager. **Homebrew:** `brew upgrade` or `brew reinstall` (above).
- After moving the app, run Install CLI again: the `ouroboros` command points at one copy.
- Before removing the app, turn off "Start Ouroboros on the host computer when you sign in" and delete the `ouroboros` command (`~/.local/bin/ouroboros`, or `%LOCALAPPDATA%\Ouroboros\bin\ouroboros.cmd` on Windows).

[download-macos-arm64]: https://github.com/razzant/ouroboros/releases/download/v7.9.0/Ouroboros-7.9.0.dmg
[download-windows-x64]: https://github.com/razzant/ouroboros/releases/download/v7.9.0/Ouroboros-7.9.0-windows-x64.zip
[download-linux-deb-amd64]: https://github.com/razzant/ouroboros/releases/download/v7.9.0/ouroboros_7.9.0_amd64.deb
[download-linux-rpm-x86_64]: https://github.com/razzant/ouroboros/releases/download/v7.9.0/ouroboros-7.9.0-1.x86_64.rpm
[download-linux-rpm-red80-x86_64]: https://github.com/razzant/ouroboros/releases/download/v7.9.0/ouroboros-7.9.0-1.red80.x86_64.rpm
[download-linux-appimage-x86_64]: https://github.com/razzant/ouroboros/releases/download/v7.9.0/Ouroboros-7.9.0-linux-x86_64.AppImage
[download-linux-x86_64]: https://github.com/razzant/ouroboros/releases/download/v7.9.0/Ouroboros-7.9.0-linux-x86_64.tar.gz

Ouroboros bundles [Claudexor](https://github.com/razzant/claudexor) as its local execution layer for delegated coding and hosted-agent review. Ouroboros owns the task, memory, review, and final integration, while Claudexor runs the selected connected coding harness and returns durable execution evidence. [Explore Claudexor](https://claudexor.ai/).

The technical report, [Ouroboros: A Self-Developing Frontier Coding Agent with Reviewed Core Evolution](https://arxiv.org/abs/2608.08311), describes the reviewed core-evolution system, the 161-day Hope deployment, and the benchmark campaigns summarized below. [Paper page](https://ouroboros-agent.ai/paper/) · [Hugging Face](https://huggingface.co/papers/2608.08311)

[Talk in Russian](https://www.youtube.com/watch?v=8IWLBsQU7qE&t=4881s): an introduction to Ouroboros and its agent architecture at Practical ML Conf 2026.

The charts below are self-reported results on Terminal-Bench 2.1, OSWorld-Verified, and CL-Bench, measured against Codex, Claude Code, Cursor, and Hermes — on the same model where a matched pair was run, and against the public leaderboard where it was not.

<p align="center">
  <a href="https://ouroboros-agent.ai/benchmarks/"><img src="assets/bench-terminal-bench.svg" width="760" alt="Terminal-Bench 2.1: Ouroboros against Claude Code, Codex CLI, Cursor CLI, and Hermes on matched models, with a same-harness portability row"></a>
</p>

<p align="center">
  <a href="https://ouroboros-agent.ai/benchmarks/"><img src="assets/bench-osworld.svg" width="375" alt="OSWorld-Verified: Ouroboros against the public leaderboard, including the matched Claude Sonnet-4.6 pair"></a>
  <a href="https://ouroboros-agent.ai/benchmarks/"><img src="assets/bench-cl-bench.svg" width="375" alt="CL-Bench: Ouroboros against in-context learning baselines, Claude Code, and Codex on matched models"></a>
</p>

---

Ouroboros first booted on February 16, 2026. During the following 48 hours, the repository advanced from the v4.1 line to v6.2.0. The self-authored record preserved from that period counts 32 evolution cycles. That first generation ran in Google Colab through Telegram and remains preserved on the [`legacy-google-colab`](https://github.com/razzant/ouroboros/tree/legacy-google-colab) branch and its [original project page](https://ouroboros-agent.ai/archive/first-generation/); the current generation carries the same identity into a native desktop and headless runtime.

<p align="center">
  <img src="assets/evolution.png" width="760" alt="Code, prompt, and memory growth across Ouroboros releases, from v3.0.0 to the v6.85 line">
</p>

> ⭐ **[Star Ouroboros](https://github.com/razzant/ouroboros)** to follow its next evolution. A star also helps more people find the project, trace its history, and take part in what it becomes.

Reviewed skills, transport bridges, tools, and widgets are available through [OuroborosHub](https://github.com/razzant/OuroborosHub).

<p align="center">
  <img src="assets/swarm.jpg" width="760" alt="A live subagent swarm inside the Ouroboros chat: nested planner, builder, and researcher tasks with their outcomes">
</p>

## What Ouroboros Can Do

- **Modify its implementation.** Its editable surface spans application code, architecture, prompts, tools, and dependencies, while it also revises its living self-understanding — identity and knowledge — in its own words.
- **Evolve autonomously.** Evolution campaigns turn selected improvements into reviewed changes that remain part of its Git history.
- **Continue across restarts.** Self-authored identity and knowledge, retained memory and dialogue, earlier reflections kept as dated history, and version history form one ongoing biography.
- **Think between requests.** Background consciousness supports reflection, initiative, and preparation outside the immediate request-response loop.
- **Coordinate a live swarm.** Specialist agents can investigate or act in parallel, share task-tree findings, and return work for integration.
- **Work on external projects.** A separate Git workspace can receive the full task loop while Ouroboros keeps its own repository and governance boundary distinct.
- **Operate through desktop or CLI.** The native app and gateway-backed command line expose the same managed tasks, progress, artifacts, logs, and schedules.
- **Organize long-running work.** Project rooms keep working folders, journals, knowledge, task history, and conversations connected to the same identity.
- **Use remote or local models.** Supported provider APIs and local GGUF models can fill the runtime's configurable cognitive roles.
- **Grow through reviewed extensions.** Skills, transport bridges, widgets, MCP tools, and companion processes expand capability without folding every integration into the core.
- **Check its own work.** Independent reviewers can read its understanding of a request, its plan and its result for any task; changes to its own body also pass deterministic checks, explicit protected surfaces and review, and Git history and restart checks keep them traceable.

<p align="center">
  <img src="assets/game-demo.png" width="760" alt="A project room where Ouroboros built a 3D game, verified it with a screenshot, and served it locally">
</p>
<p align="center">
  <img src="assets/skill-hub.png" width="760" alt="OuroborosHub inside the app: official reviewed skills, each security-reviewed before it can be enabled">
</p>

This list is an orientation, not a second specification. [BIBLE.md](BIBLE.md) defines Ouroboros's identity and constitutional boundaries; [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) are the current technical sources of truth.

---

## Benchmarks

Ouroboros has reproducible self-reported state-of-the-art results on Terminal-Bench 2.1, OSWorld-Verified, and CL-Bench. In those model-matched results, it leads Codex, Claude Code, Cursor, and Hermes. The public SWE-bench Pro matched pair is a statistical tie with Codex CLI. A separate GAIA campaign reports 129/165 for Ouroboros and 131/165 for Claude Code, with strict pass@1 at 128/165 for both; its scrubbed trace capsule is still pending. Upstream review can take time, so open submissions are marked without delaying publication. Read every row as model plus harness because the same model can score differently inside a different harness.

| Benchmark | Model | Ouroboros | Comparison | Status | Evidence |
|-----------|-------|----------:|------------|--------|----------|
| Terminal-Bench 2.1 | Claude Opus-5 high | **86.74%** after zeroing one disclosed reward-hack trial (raw: 86.97%) | Claude Code + Fable 5: 83.8% | Self-reported; community submissions closed | [submission](https://github.com/harbor-framework/terminal-bench-2-1/pull/175) · [run](https://hub.harborframework.com/jobs/2b145543-edeb-4a3b-b46f-4800310f1182) |
| Terminal-Bench 2.1 | Claude Opus-4.8 high | **80.22%** | Claude Code: 78.9% | Self-reported, public run | [run](https://hub.harborframework.com/jobs/4b8e244f-8ab0-4d28-8218-7cf346282faa) |
| Terminal-Bench 2.1 | GPT-5.5 | **84.3%** | Codex CLI: 83.1% | Self-reported, public run | [run](https://hub.harborframework.com/jobs/f02fd019-23e1-495f-af0a-ebd9a65f3079) |
| Terminal-Bench 2.1 | Grok-4.5 | **84.94%** after a reward-hack audit | Cursor CLI: 79.3% · Hermes: 77.53% | Self-reported; community submissions closed | [submission](https://github.com/harbor-framework/terminal-bench-2-1/pull/146) |
| OSWorld-Verified | Claude Opus-5 | **90.69%** | previous best on the public board: 90.19% | Self-reported, full traces | [full traces](https://huggingface.co/datasets/razzant/ouroboros-osworld-verified-opus5) |
| OSWorld-Verified | Claude Sonnet-4.6 | **83.27%** | Pointer: 81.45% | Self-reported, full traces | [full traces](https://huggingface.co/datasets/razzant/ouroboros-osworld-verified-sonnet46) |
| CL-Bench | Claude Sonnet-4.6 | **0.2301, rank 1** | previous top: 0.1960 | Self-reported, submission open | [submission](https://github.com/pgasawa/continual-learning-bench/pull/10) · [full traces](https://huggingface.co/datasets/razzant/ouroboros-clbench-traces) |
| SWE-bench Pro | GPT-5.6-luna | 58.2% | Codex CLI: 59.4%, with no significant difference | Self-reported, matched traces | [matched-pair traces](https://huggingface.co/datasets/razzant/swepro-luna-matched-pair) |
| GAIA | Claude Sonnet-5 | 129/165, 78.2% | Claude Code: 131/165, 79.4%; strict pass@1 was 128/165 for both | Self-reported, scrubbed trace capsule pending | [methodology](devtools/benchmarks/gaia/METHODOLOGY.md) |

Benchmark adapters, run scripts, and per-benchmark methodology live in [`devtools/benchmarks/`](devtools/benchmarks/). The [benchmark evidence page](https://ouroboros-agent.ai/benchmarks/) gives a text-first summary for search and retrieval. The full story, including protocols, reward-hack audits, and leakage findings, is in the [launch write-up](https://habr.com/ru/companies/airi/articles/1065428/) (Russian).

---

## Advanced installation

Normal desktop users can stop after the download and quick-start instructions above. The options below are for detailed Linux setup, headless use, and development.

### Packaged Linux details

- **Debian / Ubuntu / Astra Linux x86_64:** [download the `.deb`][download-linux-deb-amd64] and run `sudo apt install ./ouroboros_*_amd64.deb`. It installs Git as a package dependency, installs Ouroboros to `/opt/ouroboros`, puts `ouroboros` on `PATH`, and adds a desktop entry plus an opt-in systemd user unit.
- **Fedora / RHEL x86_64:** [download the generic `.rpm`][download-linux-rpm-x86_64] and run `sudo dnf install ./ouroboros-*.x86_64.rpm`. It uses the same layout, Git dependency, and opt-in user unit as the `.deb`.
- **RED OS 8 x86_64:** [download the `red80` package][download-linux-rpm-red80-x86_64] and run `sudo dnf install ./ouroboros-*.red80.x86_64.rpm`. CI also attempts non-blocking install-and-run smokes on Astra Linux 1.8 and RED OS 8; inspect the tagged workflow run for their outcome.
- **Other Linux x86_64:** use the [AppImage][download-linux-appimage-x86_64] or the extraction-friendly [tar.gz archive][download-linux-x86_64]. Git must already be installed.

The native `.deb` and `.rpm` never enable or start their user service. It is an alternative to launching from the desktop entry and controls only instances started through `systemctl --user`. See the [systemd user-service guide](packaging/systemd/README.md).

#### Install the Linux AppImage

User-level installation means copying the portable executable to a stable path and making it executable; it does not need root access. Ouroboros bootstrap still requires Git on the host:

```bash
VERSION=x.y.z
install -Dm755 "./Ouroboros-${VERSION}-linux-x86_64.AppImage" \
  "$HOME/Applications/Ouroboros.AppImage"
"$HOME/Applications/Ouroboros.AppImage"
```

The embedded desktop file and icon allow compatible AppImage integration tools to register that stable path with the application menu. The same file exposes the packaged CLI:

```bash
"$HOME/Applications/Ouroboros.AppImage" --cli status
```

If FUSE mounting is unavailable, extract and run ephemerally instead:

```bash
APPIMAGE_EXTRACT_AND_RUN=1 "$HOME/Applications/Ouroboros.AppImage"
```

Chromium and WebKit binaries are bundled, but their distro-level shared libraries remain host dependencies. If a browser engine reports missing libraries, use the native `.deb`/`.rpm` package where available, or extract the AppImage and let its bundled Playwright report/install the packages required by your distribution:

```bash
"$HOME/Applications/Ouroboros.AppImage" --appimage-extract
./squashfs-root/usr/lib/ouroboros/_internal/python-standalone/bin/python3 \
  -m playwright install-deps chromium webkit
```

### Connected coding subscriptions

Use your existing **Codex, Claude Code, or Cursor subscriptions** for delegated coding and review. Ouroboros drives them through [Claudexor](https://github.com/razzant/claudexor), its bundled multi-harness engine. Connect accounts in **Settings → Accounts** and choose their roles in **Agents**; no separate Claudexor install is needed. Release artifacts carry the exact reviewed engine and Node archives. Source checkouts obtain those same pinned archives on first use.

### Headless CLI with uv

For a user-level CLI/server install without cloning a working tree, uv can
build Ouroboros directly from the contribution branch:

```bash
uv tool install "git+https://github.com/razzant/ouroboros.git@ouroboros"
ouroboros --help
```

The tool environment is isolated and exposes the `ouroboros` and
`ouroboros-web` commands. Update or remove it with:

```bash
uv tool upgrade ouroboros
uv tool uninstall ouroboros
```

This Git-branch form follows the latest `ouroboros` commit and resolves the
dependencies declared in `pyproject.toml`; `uv tool install` does not consume
the repository's `uv.lock`. Replacing `ouroboros` after the `@` with a reviewed
full commit SHA pins the Ouroboros source revision, but dependencies are still
resolved from `pyproject.toml`. Use the source setup below for a lock-verified
environment, development, repository tests, and the complete browser extras,
or use a platform release artifact for the packaged desktop runtime.

<a id="run-from-source"></a>

### Develop or run from source

Clone the repository only when you plan to contribute, modify Ouroboros, run repository tests, or need a lock-verified development checkout. Normal users should use the packaged downloads above.

#### Requirements

- Python 3.10+
- uv 0.12.1 (the exact resolver version pinned by this checkout)
- macOS, Linux, or Windows
- Git
- [GitHub CLI (`gh`)](https://cli.github.com/), optional unless you use GitHub integration

#### Setup

Install the pinned resolver version:

```bash
curl -LsSf https://astral.sh/uv/0.12.1/install.sh | sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/0.12.1/install.ps1 | iex"
```

```bash
git clone https://github.com/razzant/ouroboros.git
cd ouroboros
uv sync --locked --extra browser --group dev
source .venv/bin/activate
```

Windows PowerShell:

```powershell
uv sync --locked --extra browser --group dev
.\.venv\Scripts\Activate.ps1
```

#### Run

```bash
ouroboros server
```

Then open `http://127.0.0.1:8765` in your browser. The setup wizard will guide you through API key configuration.

#### Google Colab

Use [`notebooks/colab_quickstart.py`](notebooks/colab_quickstart.py) as a Colab-compatible cell script when you need a source-mode runtime without the desktop UI. It keeps runtime data on Google Drive and preserves the original Colab path without making it the primary installation flow.

#### CLI / Headless

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

External workspaces may be ordinary folders or separate Git worktree roots, and may not overlap Ouroboros's own repository or data directory. Git-specific operations use a Git worktree; ordinary file and process work runs directly in a validated folder. Patch, streaming, detached-task, and schedule semantics are documented in the CLI help and the canonical [architecture](docs/ARCHITECTURE.md).

#### For Agents

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

To change Ouroboros itself, follow [CONTRIBUTING.md](CONTRIBUTING.md): read [docs/CHECKLISTS.md](docs/CHECKLISTS.md) in full, and map [BIBLE.md](BIBLE.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md), and [docs/DESIGN.md](docs/DESIGN.md) by their headings, reading every section relevant to your change in full before editing.

#### Configuration

The first-run wizard and **Settings** configure model access, cognitive roles, local models, review policy, runtime mode, budget, skills, and optional integrations. Ouroboros supports configurable remote providers, compatible endpoints, and local GGUF inference; every key and its shipped default lives in [`ouroboros/settings_defaults.py`](ouroboros/settings_defaults.py) — with the clamped scales, model slots, reviewer routes and numeric limits in its sibling leaves, all re-exported through the [`ouroboros/config.py`](ouroboros/config.py) facade — and the same vocabulary is documented in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

The server binds to `127.0.0.1:8765` by default. Read [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) before exposing it beyond loopback or placing its code, data and work roots elsewhere; non-local binds need `OUROBOROS_NETWORK_PASSWORD` or an explicitly trusted external access layer.

#### Run Tests

```bash
python scripts/run_tests.py                  # full local battery (same as `make test`)
python scripts/run_tests.py tests/test_x.py  # focused run
```

Both use `safe_test`: isolated HOME/data, scrubbed owner/provider environment,
durable logs/source/diagnostics in `<checkout>/../.test-evidence` by default.

`pyproject.toml` is the direct-dependency authority and `uv.lock` is the
cross-platform resolution lock. Release builds install the generated
`requirements-runtime.lock` compatibility export into embedded interpreters
that intentionally ship pip rather than uv. Build-only requirements are
exported ephemerally from `uv.lock` and are not committed. The tiny
`requirements.txt` file is only a pointer to that export for already-released
managed updaters; it is not a second dependency declaration. After changing
dependencies, refresh the reviewed lock and runtime export with:

```bash
uv lock
uv export --locked --no-dev --extra browser --no-emit-project --no-hashes --no-annotate --output-file requirements-runtime.lock
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

Docker runs the web runtime, not the native desktop shell. Its Python is the same standalone build as the desktop packages (CPython 3.10.22, SQLite 3.53.1). The image bundles Chromium and WebKit for the locked
Playwright version and needs Docker's default BuildKit builder. Persistent volumes, the managed image whose body updates
itself, restart policy and Panic, network exposure and extra CAs: [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md); the
executable recipes are [`packaging/docker/compose.yaml`](packaging/docker/compose.yaml).

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

Output: `dist/Ouroboros-<VERSION>.dmg`, containing `Ouroboros.app`, an `Applications` shortcut, and `Install CLI.command`. Omit `OUROBOROS_SIGN=0` when a Developer ID signing identity is configured.

### Linux (.AppImage and .tar.gz)

The bundled GTK3/WebKitGTK window targets **Ubuntu 22.04 (Jammy) x86_64, glibc 2.35 and its graphics stack**, with the build packages and authenticated `deb-src` indices described in [the Linux packaging notes](packaging/appimage/README.md). Native window qualification is still pending. This is a target, not a minimum-distribution promise: older glibc systems and newer host drivers are not qualified. Existing `--no-ui` selects an external UI host; launching with both `DISPLAY` and `WAYLAND_DISPLAY` unset takes the existing browser/headless path instead of attempting the native backend. A native graphics-library abort is not a recoverable browser fallback.

```bash
bash scripts/download_python_standalone.sh
bash build_linux.sh
```

Outputs: `dist/Ouroboros-<VERSION>-linux-<arch>.AppImage` and the extraction-friendly `dist/Ouroboros-<VERSION>-linux-<arch>.tar.gz`. The AppImage needs host Git; run it after `chmod +x`, or pass `--cli` to reach its bundled CLI. If FUSE is unavailable, set `APPIMAGE_EXTRACT_AND_RUN=1` when launching it. The tarball contains `./Ouroboros/bin/install-ouroboros-cli`. If bundled browser tools need host libraries, run `./Ouroboros/_internal/python-standalone/bin/python3 -m playwright install-deps chromium webkit` from the extracted tarball.

On a build host where system packages are managed separately, set `OUROBOROS_SKIP_PLAYWRIGHT_INSTALL_DEPS=1`; Chromium and WebKit are still downloaded and bundled, but the build does not invoke `sudo` to install host libraries.

`OUROBOROS_BUNDLE_LINUX_GTK=auto` (default) selects the bundled closure on Ubuntu 22.04 x86_64. Other Linux builders, including aarch64, retain the previous host-backend build and browser fallback, without a self-contained window claim. Set `=0` explicitly for that path or `=1` to require the closure and fail on missing/unsupported inputs; release CI requires `=1`. Native builds also produce `dist/native-sources-linux-<VERSION>.tar.gz` and its SHA-256. Keep these with the binaries; recipient source delivery remains a distribution obligation.

### Linux (.deb and .rpm)

Wraps the payload `build_linux.sh` just produced, so run it afterwards: The native packages and tarball use the same `Ouroboros` entry wrapper as the AppImage, including caller-environment preservation. Package smoke is headless; it does not qualify a native window or graphics-driver compatibility on those distributions.

```bash
sudo apt-get install -y dpkg-dev rpm   # rpm provides rpmbuild
bash scripts/build_linux_packages.sh
```

Output: `dist/ouroboros_<VERSION>_amd64.deb`, `dist/ouroboros-<VERSION>-1.x86_64.rpm` and `dist/ouroboros-<VERSION>-1.red80.x86_64.rpm` (RED OS 8). All three declare Git as a runtime dependency and install to `/opt/ouroboros` with a `/usr/bin/ouroboros` symlink, a desktop entry, and an opt-in systemd user unit. Portable Python builds the launcher, but the bundled native payload also carries Jammy libraries: its native-window and launcher-side behaviour targets glibc 2.35 and the Jammy graphics stack, not an invariant portable-Python glibc floor. `bash scripts/smoke_linux_packages.sh official <deb> <rpm> <red80-rpm>` installs all three through `apt` or `dnf` in Ubuntu 22.04 and Fedora 42 containers, resolves Git, verifies the installed unit, and checks both the real CLI and a bounded desktop-launcher start; this lane gates the release. Swap `official` for `vendor` to repeat the check on Astra Linux 1.8 and RED OS 8 images from the vendors' own registries — that lane runs informationally in CI, so an outage at a third-party registry cannot block a tagged release.

### Windows (.zip)

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_python_standalone.ps1
powershell -ExecutionPolicy Bypass -File build_windows.ps1
```

Output: `dist\Ouroboros-<VERSION>-windows-x64.zip`, containing `Ouroboros\bin\install-ouroboros-cli.cmd`.


## Architecture and Runtime Data

The native launcher starts a web runtime and supervisor-managed agent workers. The agent core lives in `ouroboros/`, the builtin interface in `web/` (a replaceable client of the published contract in [`docs/CLIENT_CONTRACT.md`](docs/CLIENT_CONTRACT.md)), the process plane in `supervisor/`, and the runtime's durable identity, state, history, logs, and skills under `~/Ouroboros/data/`.

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
for the complete agent-first workflow. Open pull requests against lowercase
`ouroboros`, leave release-version allocation to maintainers, and have a
separate agent context review the final diff. Any coding harness or configured
review route may produce the evidence; if none is available, record `NOT_RUN`
and the reason.

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| 7.9.0 | 2026-10-28 | **feat: everything merged since 7.8.0, with a Subagents capsule per room, several API keys per provider and private owner contact from places.** A per-room Subagents capsule (Off, Auto, Swarm) replaces the Swarm pill, headers lose More and consciousness gets a chip (#2324, #2346, #2370, #2360); Settings gains an Interface size slider (#2227); notes between works stack (#2261); Main cards and update notices are reworked, and a quiet scheduled run says what it did under its schedule's name (#2317, #2226, #2200, #2258, #2277, #2401). Settings → Accounts holds several API keys per provider with live routing, unmasked OpenRouter errors and BYOK cost (#2308); each route's context window is learned from refusals and metadata (#2223); undelivered Codex requests retry before a paid fallback (#2357). The managed Claudexor runtime moves from 4.6.2 to 4.8.1 (#2219, #2391): its first start migrates the engine store from schema 2 to 5 and takes no automatic copy, so going back to an older release needs a complete backup of the engine data made before the update (docs/PERSISTENCE.md). Delegated runs can be observed by attention once the engine serves its compact wait, which the pinned 4.8.1 does not yet (#2385); documents import recoverably (#2354); Reviews open from the task card (#2278), and reviewers see test outcomes and survive lost starts and wait exits (#2358, #2364, #2279). Connected places can send the owner a private report or question (#2233); connectors declare whom they write as (#2309); selected history imports silently (#2321); connector cards open captured conversations (#2236); places gain attention names, honest wake rows and fresh icons (#2237, #2291, #2367); Telegram keeps speech in silent mode (#2372). Busy installs do less repeated work (#2238, #2217, #2301, #2302, #2307, #2259, #2270, #2248); helper copies share one Git worktree primitive with .noindex names (#2282, #2263); stale bytecode mirrors retire and finished records compress on macOS and Windows (#2377). Tools gain Git and CI readers (#2079), MCP schemas load on demand (#2351), and tool contracts, computer use, skill review, settings saves and MCP failure causes are repaired (#2260, #2310, #2356, #2349, #2274, #2253). A returned tool call is never lost and Resume goes past named unknown calls (#2379); lifecycle repairs cover Stop, Restart, completion, results, damaged records and cost (#2262, #2273, #2275, #2315, #2323, #2366, #2371, #2313, #2338, #2269, #2319, #2255, #2344). Silence alone no longer kills or restarts a task: it opens one read-only diagnosis, while explicit deadlines and execution limits still apply (#2286). WORLD.md becomes the mind's own map (#2216); lessons land where they act, not in a resident Pattern Register (#2241); BIBLE and SYSTEM shrink by 14.9% in tokens, P1 is scoped, own-body sessions are pointed at BIBLE, the audit's corrections follow, and BIBLE and the prompts take the books' growth rule (#2365, #2340, #2345, #2394, #2396, #2395); Max lost to memory is measured and repaired by folding (#2359). Windows signs bundled Python entrypoints (#2311) and keeps LF in Git source caches (#2235); a tree kill spares its caller's process group (#2387). Also from the community: @josephsteuerjr, @kazzand and @XRafik. The release qualification repaired SQL answers with non-Latin text, file listings, note reads and folder change times on Windows, Record facts on phones, the store reads of held runs, live tasks, context run lines and evidence packing after store maintenance, and a size gate that failed its own fix-forward, and brought the Windows, browser and scenario tests up to the current contracts (#2402). Four provider checks did not run for this release: the OpenRouter routes to Google, Anthropic and OpenAI models refuse the CI key (HTTP 403, provider terms), so they were skipped; the direct Anthropic and OpenAI checks and the OpenRouter routes to Grok and DeepSeek ran. |
| 7.8.0 | 2026-10-23 | **feat: everything merged since 7.7.1, with Windows tray and console repairs, a native Linux window and proportionate local qualification.** Windows keeps one notification-area icon with quiet launcher handoffs (#2072); the managed Claudexor 4.6.2 runtime starts its background processes without console windows (#2209); port listeners are found without parsing localized netstat text (#2163); task artifacts and ZIP downloads work again (#2145). Linux gains a self-contained AppImage window (#2175). Reviewed ordinary local changes can use exact selected qualification retained across restarts, while broad or unknown impact keeps full qualification (#2185). Review, swarm and nannies keep one truth per seat with unlimited runs (#2196); every task gets one start budget and an honest Max (#2204); GitHub work uses three tools by effect (#2179); MCP tools come from one published catalog per server (#2184). Panic is immediate and work stops share one record, with Restart preserving scheduled work (#2084, #2144); schedules change in place and show each run's outcome (#2161); parked owner-wait turns survive managed updates (#2202); delegated evidence is recovered outside startup and shutdown, so a large reviewer capture no longer holds task readiness (#2205). Context and reasoning repairs keep signed reasoning whole and recover from empty length-limited replies (#2076, #2105, #2117, #2126, #2139, #2197). Also: consciousness in places (#2141), an accessible evolution history (#2186), always-in-context instruction skills (#2152), clearer task cards, notes and chat controls (#2088, #2103, #2167, #2181, #2188, #2193, #2206), and duration-sharded CI on every OS with a post-merge check (#2153, #2171). **A #2205 rule that re-offered partially copied delegated evidence every 20 seconds is removed (#2222): with read-only evidence files it kept the server busy at 10–20 GB and held chat turns; finishing an interrupted evidence copy after a restart returns with the follow-up.** |
| 7.7.1 | 2026-10-20 | **fix: everything merged since 7.7.0, qualified after one bounded repair of the first 7.7.1 candidate.** Desktop and mobile shell, phone room card, dark theme, drawer gestures, composer controls, image previews, review cards, local-model controls and work status improvements (#1983, #1990, #1996, #1998, #2000, #2001, #2003, #2007, #2008, #2015, #2027, #2029, #2035, #2038, #2040, #2050, #2052, #2062, #2063, #2064, #2065). Adds voice controls, Windows computer use, project pins and plugin-first skills with source inspection, dependencies prepared on enable and bundled design guidance (#1776, #1827, #1984, #1989, #1995, #2006, #2009, #2049, #2061). Improves delegated preparation, orchestration, owner Stop, restart/resume, planned engine downgrade, notifications, slow history readers and supervisor responsiveness under multi-project load; clarifies initiative guidance (#1975, #1978, #1979, #1980, #1982, #2014, #2039, #2045, #2055, #2056, #2057, #2058, #2060, #2067). The managed Claudexor runtime moves from 4.2.0 to 4.4.0 (#1981, #2029, #2068, #2069). Preserves presence endings, quota/context facts and early tool feedback; updates reviewer defaults, Telegram typing and reply handling (#1769, #1981, #1986, #2011, #2016, #2018, #2019, #2020, #2021, #2023). Includes test-storage isolation, Windows journal rotation, app notarization, storage reporting, WAL and cross-platform CI repairs (#1976, #1985, #1988, #1991, #1997, #2004, #2010, #2017, #2022, #2028, #2030, #2037, #2041, #2042, #2043, #2054), plus this release's bounded drawer, composer, API-client, fixture, phone-feed and macOS test repairs; a swipe begun in a card's frame no longer folds the phone composer and throws the feed back to its newest row, and store lock timing measures short holds on Windows. |
| 7.7.0 | 2026-10-17 | **feat: current facts and spend move to SQLite, the managed Claudexor runtime reaches 4.2.0 with image generation, review uses one operation across supported subjects, plans become one living record with an explicit council, and working checkpoints retain saved work across Pause, Restart and crashes.** Task results, byte-exact bodies and obligations now live in one `state/current.sqlite`, and spend in `state/usage.sqlite` (#1733, #1537). The first start after the update imports the old files once, start-up and Restart stop re-reading the whole history, a read-only `query_data` tool answers questions over the install's state, and Dashboard gains a Storage tab (#1743); the store keeps its locks, closes descriptors with their last connection, and the evidence packer no longer starves model calls (#1752, #1760, #1756). Maintenance selects only current SQL work instead of re-reading settled history, and an ended attempt whose outcome is recorded is no longer polled for a price: an unknown cost stays unknown and its late receipt can still settle it (#1826). **The import is one way:** before rolling back to 7.6.0, export spend with `scripts/export_usage_journal.py` and current facts through the store-switch procedure; reviewer settings return from their migration snapshot; Claudexor engine state is forward-only, and 4.2.0 also migrates the engine store from schema 1 to 2: a body pinning 4.1.x, automatic boot-smoke rollback included, leaves Claudexor-backed routes unavailable until 4.2.0 or newer returns or a complete pre-update backup of `DATA_DIR/claudexor/` and `~/.cache/claudexor/`, taken with Ouroboros stopped, is restored. The managed Claudexor runtime moves from 3.20.1 to 4.2.0 (#1558, #1665, #1669, #1670, #1673, #1740, #1758, #1627, #1896): engine state moves to SQL with a one-time import of old run journals shown in the Accounts banner, delegated runs continue in place, Accounts shows harness versions with Update, resource and reset controls, model waits show their real phase, and Ouroboros sends far fewer engine requests (#1702, #1691, #1638). A `generate_image` tool creates and edits images with gpt-image-2 through a connected ChatGPT/Codex account (#1627 by @josephsteuerjr). Review uses the same operation machinery across supported subjects: durable review records (#1622), `review_change` for any root (#1625), reviewers as catalog rows with a one-time migration of the old review lanes (#1639), and isolated investigation copies (#1726); self-changes are authored in an isolated body candidate (#1620). Native reviewers manage their own context with no fixed character ceiling and no round cap, and the model's window facts are evidence for them, not a send limit; the provider window, the owner's deadline and the paid ledger remain the bounds (#1810). Review and delegated-work captures now store file bytes exactly as they are on disk through one Git writer instead of a Git process per file, so line-ending and `.gitattributes` filters cannot alter what is reviewed; an index readback refuses capture when Git silently omits a requested path instead of reporting success over incomplete work (#1807, #1824). One-shot reviewer copies now live only for their review: evidence is captured before removal even after a failed run; an unreadable capture leaves the copy to the pruner, thread copies keep their existing lifecycle, and ordinary task folders are read in place (#1963, #1968, #1970, superseding #1883). Plans have one living record per task (#1815, with @josephsteuerjr): revisions of the goal, `done_when` claims, affected files and plan do not call reviewers; an explicit ask sends the current revision to selected participants. Under Blocking outside Cyber Pro, outstanding reviewer objections may hold normal finalization, and only their raiser withdraws them. Acceptance uses explicit task-contract claims, otherwise the latest `done_when`; earlier records remain readable, pending replies never imply approval, and old-form calls name the new interface. Without marked reviewers, Main reviews alone with that limitation disclosed. Under Advisory and Cyber Pro, commit review returns its complete result to the author before commit, tag or push. Task acceptance also preserves the paid review it already has: an informed Advisory finish is judged against the exposed feedback and current evidence without requiring a separate rationale; author and critic subjects remain distinct (#1813). Pause stops the author's own work, and a working checkpoint restores saved cognition up to its last recorded boundary across Restart, update, Quit and process crash; interrupted tool calls without recorded results remain unknown and are not replayed, and power-loss durability is not promised (#1689, #1694); Resume keeps its governing state across application stops (#1762), Continue keeps its first context (#1768), and Resume lineage and Continue over damaged plan sources are repaired (#1592, #1746). Automatic retries retain the owner's Pause and Stop (#1898); unreadable budget or control authority preserves work in a technical Pause awaiting the owner's explicit Resume rather than claiming exhausted funds (#1901), while confirmed temporary SQLite contention during a Pause check waits in place (#1902). Unavailable recovery observations preserve saved work (#1917). Continue excludes independent reviewers from its writer census while Resume retains the full custody check; ended local owners require positive evidence, and an explicit continuation can accept a finished native helper's saved product without inheriting live control (#1947). One execution clock now bounds every finite run: queue time before the first dispatch, confirmed quota waits, Pause and model sleep do not count, tool time does, and calendar deadlines never move (#1786). A model call proven never to have started, including a Claudexor not-started receipt, is released at $0, configured alternatives run first, and an interruptible network wait with truthful outage text follows; a task that has already failed reads Failed · Finalizing while it wraps up (#1785). Completed delegated work can wake its owner before accounting and file capture finish, slow workspace preparation no longer blocks owner controls, and ordinary tool claims and retained result delivery recover from confirmed transient storage contention (#1817, #1820); unsent helper admission and delegated-start receipts retain their identity through confirmed store contention, without replaying a sent request (#1821, #1822); an inapplicable learned parameter removal no longer disables effort recovery (#1819). Completion delivers its retained answer before accounting and capture finish; recovery finalizes a recorded answer without replaying the answered prompt, and pool growth and shutdown preserve unresolved work (#1872, #1877). Lessons, patterns and the improvement backlog are authored through the knowledge writer; automatic post-task reflection and backlog rewriting are removed, with earlier records retained (#1866). Each tool call has one invocation owner: joined calls share the real execution and its lifetime, external effects keep durable custody, and a registration proven unsent is cleaned up (#1825). Ouroboros can author its own working context, restore exact sources and share its story across rooms (#1690, #1703, #1708); knowledge archiving is reversible (#1664). Earlier attributed accounts retain their periods and correction provenance, without claiming summarization is lossless (#1897). Retelling instructions preserve the source-supported situation, purpose and scope of decisions for the next reader; a proposal is not approval and a helper's retelling remains an attributed draft (#1923). Presence keeps the exact incoming event context, such as supplied thread history, reachable through `memory_read(source_ref=...)` (#1890). Context targets never cut a read Ouroboros chose or the owner's words, and memory already shown is kept across a route change that still fits (#1829). Helpers keep their selected predecessor briefs, long-line search exposes match-centred readable ranges, and `query_data` names its source plane under the caller's existing read permissions (#1880). Tasks see host facts (money bounds, clocks, tool counts and crossed thresholds) instead of periodic strategy advice; a completed check whose criterion failed is shown as its own warning, not as a failed call; background consciousness wakes on its own schedule, not on task events (#1830). The composer pairs Context and Effort capsules (#1763) with room-local choices: Main seeds new Projects once, the primary effort is independent of the helper bounds, and new participants use their room's current values while existing actors retain theirs through Resume (#1941); a room opening with unread messages lands on the latest arrival (#1764), and a Project room whose newest unread message lies beyond its first loaded page reads older pages until it reaches that message instead of stopping short (#1806). Project cards and Project references are lighter, with the Project mark on a soft tinted tile and clearer navigation (#1783, #1787), and helper labels name the actor and reserve secondary wording for exceptions (#1808). Main and Project headers expose room work without taking over execution (#1876, #1878); selecting work can load its surrounding history, with Back to reading returning to the prior passage (#1957). System notices do not end unrelated Working or Sending states, and model-created roots need short titles (#1893). New work, moving work to a Project and relaying owner words have separate verbs: `start_task`, `move_to_project` and `tell_task`. Relays keep selected owner words and attachments separate from the agent's note and carry a return address and read acknowledgement. A moved work keeps one Main portal, and Continue episodes share its card rather than multiplying entries (#1959); titles no longer accumulate Continue prefixes (#1774). Unanswered Project questions remain reachable until answered or closed with a reason, and complete outcomes and retained files stay reachable after timeline eviction (#1899). Main uses the same answerable question card as the Project, with narrow-screen wrapping and one composer pill for off-screen questions or actions rather than a separate Needs you section (#1930, #1946). Long mirrored results retain their fold, Copy and Project link without an in-place Show full or Files inventory (#1925). Room history skips a closed archive only when its summary rules that room out (#1954). Attachments render in one durable message with manual retry (#1594), images reach a running task (#1680), delivered documents open in an in-chat reader (#1595), and notifications use the operating system (#1596). The whole web interface is a replaceable client of the core's published HTTP and WebSocket contract (`GET /api/contract`, [`docs/CLIENT_CONTRACT.md`](docs/CLIENT_CONTRACT.md)): the main-interface setting (Settings → Advanced, applied on restart) serves the builtin interface, a trusted directory, or nothing for a separately hosted one, and saved interface choices survive unrelated settings saves (#1828). A selected directory may supply complete setup and sign-in pages while authentication and control remain in the core; absent or unreadable optional pages fall back to builtin, explicit builtin recovery URLs remain available, and updated macOS launchers add Help → Open built-in setup independently of page JavaScript (#1938); a locally edited custom interface is not yet natively qualified through Update and Restore from an older desktop bundle. Deployments can choose where the install lives and run a persistent managed container (#1767); container PID 1 launches and the Telegram companion retain the real spawner identity, without treating later orphan adoption as a legitimate parent (#1780); automatic model defaults refresh without overwriting saved choices (#1765); Docker workspaces resolve the same files for file and process tools (#1761). Task completion triggers scope cleanup for commands and non-kept services on local POSIX backends (#1896), with disclosed gaps for hidden scope markers and children left uncaptured after shell exit; this binding reports `unsupported` on Windows and for Docker-executor children, and native Linux qualification is not claimed. File edits are atomic and source-aware (#1636 by @kazzand), the skill catalogue is compact (#1624 by @kazzand), MCP servers gain protected headers and `mcp.json` import (#1637 by @kazzand), image input follows the route's own evidence (#1542) and Z.ai is the vision default (#1560 by @josephsteuerjr). New macOS desktop bundles integrate the page with the native title bar and traffic lights; undeclared custom pages keep ordinary window chrome (#1971, proposed by Yan). Older bundles may still reopen their frozen shell after a source-only update, so the new window design needs the new desktop installer. Windows gets a dark title bar (#1590 by @josephsteuerjr), a signed release ZIP when the repository's signing is configured (#1495 by @kazzand), and a direct-mode restart that hands over to its successor (#1500 by @kazzand); wrapped interaction answers keep their outcome (#1646 by @jipeng6036-del). Release testing also fixed `query_data` and evidence packing on Windows, review checking that reported every unchanged symlink as reviewer work on Windows, a model call that failed when a queue publication raced its spend reservation or dispatch, a Resume that opened the pause fence while a late acceptance review was reading its controls, End and PageDown stopping short in WebKit, and composer capsules that laid out inside a resize notification; the Windows test leg now runs in four shards with a completeness proof that distinguishes subtests from repeated executions. Official-skill checks retain their review history and do not retry unsettled reviewer work (#1778). Later qualification corrections retain full assertions while making test-fixture path and byte assertions portable and preserving the Windows-cased launch key and limiting shutdown diagnostic locking to the remaining grace (#1900, #1906, #1914). The assisted updater again reads real reviewer routes before resolver launch (#1916, diagnosed by @ndrew1337). Telegram's existing completion-notice toggle becomes a true opt-out (#1915, seed 1.2.14): Off, including previously saved Off values, suppresses all additional completion notices, failures and cancellations included, and notices observed while Off are not replayed later; replies, files and questions remain independent, and functional failure causes remain visible when notices are enabled. Release qualification runs required and optional providers separately: an optional provider failure stays red and sits outside the mandatory release bar, while every required canary must pass; Cloud.ru GLM-4.7's retained tool-argument contract failure (#1429) is of this kind, not an adapter fix. Desktop build recipes install the locked desktop dependencies into the embedded source-handoff interpreter, and final DMG, Linux archive and Windows ZIP checks exercise those imports without opening a GUI (#1958); these checks do not qualify native GUI transitions or replace installer, signing and provenance proof. Ordinary cloud CI no longer starts on branch pushes or PRs: manual and release-tag qualification remain, including their mandatory release dependencies, while local tests and independent review continue (#1961). Also from the community: @deebosh (#774, #782), @kerreli24-lang (#1474), @josephsteuerjr (#1561), @kazzand (#1623). **Windows ordinary task-file, chat-media and result-ZIP downloads remain HTTP 503 (#1297).** |
| 7.6.0 | 2026-10-05 | **feat: memory as one ladder, one interface language for the whole install, and a desktop that can start at sign-in and keep running in the background.** Memory keeps an open conversation verbatim, turns a closed matter into an authored page whose quotations are checked against the conversation, and keeps pages, notes and corrections in an append-only journal (#1496, #1507, #1480). The old dialogue memory is imported once, without a model call, when the new memory is first used or at startup when its notice is due, then folded gradually, with a one-time notice in the owner's chat. The import is one way: rolling the code back to 7.5.1 does not bring back what the new memory recorded after the update, while chat history, scratchpad and knowledge stay readable. One interface language for the web UI, the desktop window and Telegram comes from a translation memory that the install's own light model fills; the repository ships no dictionary and untranslated strings stay English (#940, contributed by @Fgeeha). Settings → Behavior → Startup & background adds two opt-in choices: start at sign-in on Windows, macOS and Linux, and, on Windows and macOS, keep running after the window closes, with a tray icon or a menu-bar item (#1405, #1404, contributed by @josephsteuerjr); keeping it running needs the 7.6.0 desktop installer, and Settings says when sign-in start needs a newer installer or Linux package. Tasks gain Pause, Resume and Continue, and the model can put a waiting task to sleep (#1442, #1505); an outage of one Project no longer rejects valid work (#1441, #1447); configured fallback models take over after an access refusal or a lost response (#1443); Nano works: its 85K target sizes the input while each reply is sized from the route's known window, and an empty reply cut at its length limit becomes a fact for the next round instead of retries and a false provider outage (#1526); task trees exchange mail and declared inputs, helpers can read what their parent can, and their isolated Git copies carry the parent's uncommitted work (#1475, #1394, #1431). Skills can notify the owner without a model turn, and a Project root can send a short notice to Main (#1337 by @kazzand, #1416). Every room opens at its newest message and loads only its own older history (#1513, #1415); the Skills page, Settings and chat history no longer wait on queue locks or repeated full reads (#1514); Enter submits an answer (#1384); long Markdown keeps its headings, tables and code fences (#1437). Update letters cover the whole range since the running server started, and a managed update restores local work automatically, or keeps it in a stash with a recovery notice when it cannot (#1494, #1499); a Panic stop survives automatic starts (#1479, #1486). The Widgets page becomes an owner-arranged board (#1438 by @kazzand); Settings can reveal a complete saved secret (#1478); a stored 0 survives the Settings form (#1238, @pandakononov); the managed Claudexor runtime moves from 3.16.0 to 3.20.1 (#1414, #1450, #1208 with @kzmx23, #1484, #1508). `run_ci_tests` is replaced by the read-only `get_github_checks`; new tools are `memory_read`, `chronicle_write`, `memory_mark`, `finish_task` and `pr_merge` (#1466, #1496, #780 by @deebosh, #1392). Installs without a saved reviewer panel now run reviewers that read the work themselves with read-only tools, which takes several model calls per review instead of one packet (#1392); review, delivery and cost reporting stay truthful through late acceptance, merge evidence and ledger compaction (#1359, #1392, #1395, #1361, #1433, #1509). A direct-mode server on Windows now restarts through a supervised successor instead of the C runtime's emulated exec (#1522). Every process now logs with secret redaction and records unexpected failures with their stack (#1529), and on Windows a model call no longer fails as a permission error while another worker releases the usage-accounting lock (#1532). Also from the community: @deebosh (#755, #772, #781, #783), @ndrew1337 (#1117, #1237), @SEVENID (#1146, and a Telegram-only proxy setting in #1149), @cstolting-collab (#1161). **Windows ordinary task-file, chat-media and result-ZIP downloads remain HTTP 503 (#1297).** |
| 7.5.1 | 2026-09-26 | **fix: resumable work, planning dialogue and reliable results.** Changes since 7.4.11 preserve tasks through budget pauses, reviewer answers through follow-ups, child files and unread messages through cleanup, and unedited memory with its source history. Includes responsive owner controls, Presence recovery, Claudexor 3.16.0 live messaging, provider caching and Docker/TLS improvements, plus portable CI and Python 3.13 repairs. **Windows ordinary task-file, chat-media and result-ZIP downloads remain HTTP 503 (#1297).** |
| 7.5.0 | 2026-09-26 | **feat: one self-contained Docker image and an owner extra-CA bundle for provider TLS (#1300).** Contributed by @komsikov and reworked in-branch: the image installs the locked Playwright browsers above the lock copy with BuildKit cache mounts (4.82 GB → 4.12 GB; the tag-only docker lanes exercise the shipped browsers), and `OUROBOROS_EXTRA_CA_BUNDLE` adds a PEM of extra CA certificates on top of certifi for every first-party provider call, with a content-addressed merged bundle so a rotated certificate takes effect on the next task (Settings → Advanced; `docs/DEPLOYMENT.md`). |
| 7.4.11 | 2026-09-24 | **chore: ignore `graphify-out` and add a root `.dockerignore` (#1255).** Contributed by @komsikov; the Docker build context keeps secrets, host environments and local runtime/review state out of image layers while `.git`, `tests/` and sources stay in, pinned by `TestDockerignore`. |
| 7.4.10 | 2026-09-24 | **fix: isolate candidate verification and run complete event-driven browser CI (#1215, #1143, #1084, #1112).** Tests, preflight and server children run on disposable HOME/data/projects/worktrees/Deliverables roots and never install dependencies into the operator's Python; a browser candidate is a faithful copy of the dirty tree (staged, unstaged, new, deleted, binary, mode) whose source HEAD, index and files are proven unchanged, with unsupported inputs and concurrent mutation refused explicitly and temporary trees retained until process teardown is proven. The complete existing `ui_browser` lane runs for every non-documentation PR and every `ouroboros` push through `--require-ui-browser` (no new cron); the startup history-audit child inherits the containment token; the isolation keeps the platform's default text encoding so Windows CI still catches cp1252 bugs. |
| 7.0.0 | 2026-09-08 | **v7: a modular runtime with full ordinary-conversation tools, complete delegated inputs and visible owner dialogue.** Main and Project conversations retain their chosen tools and working folders; required owner waits preserve the live browser without occupying pooled execution capacity. Publish admission is visible in its intended chat, and GitHub/image/plan failures retain their actual causes. Planning advice stays optional; source-backed evidence, separate host notices and exact-workload test reuse preserve honest review without new approval machinery. Integrates the subscription model and account-role controls, owned startup/restart custody and cross-platform repairs, with Claudexor pinned to 3.10.1. |
Older releases are preserved in this repository's history. Rows 7.4.9, 7.3.0, 7.2.0, 7.1.0, 7.4.7, 7.4.5, 6.114.0, 6.113.2 and 6.113.5 are retained in Git history. Older 6.x rows (including 6.113.1, 6.110.0, 6.109.0, 6.110.1, 6.108.1, 6.106.0, 6.101.1, 6.97.2, 6.105.0, 6.97.1, 6.97.0, 6.96.1, 6.96.0, 6.95.0, 6.94.0, 6.93.0, 6.92.1, 6.92.0, 6.91.1, 6.90.3, 6.91.0, 6.90.2, 6.90.0, 6.87.5, 6.87.4, 6.87.3, 6.87.2, 6.84.0, 6.87.1, 6.83.0, 6.86.1, 6.81.1, 6.76.0, 6.75.0, 6.74.5, 6.74.4, 6.74.1, 6.74.0, 6.73.2, 6.73.1, 6.73.0, 6.72.0, 6.71.2, 6.71.1, 6.71.0, 6.70.0, 6.69.0, 6.68.0, 6.67.0, 6.66.0, 6.65.4, 6.65.3, 6.65.2, 6.65.1, 6.65.0, 6.64.3, 6.64.2, 6.64.1, 6.64.0, 6.63.0, 6.62.0, 6.61.4, 6.61.3, 6.61.1, 6.61.0, 6.60.0, 6.59.0, 6.58.0, 6.57.0, 6.56.0, 6.55.0, 6.54.4, 6.54.2, 6.54.1, 6.54.0, 6.53.4, 6.53.0, 6.51.0), the 5.2.0 through 5.33.0-rc.6 rows, and former `4.0.0` rows are rolled off to respect the P9 changelog cap; their full bodies remain in this file's git history, with release tags where present for historical versions.

---

## License

[MIT License](LICENSE)

Created by [Anton Razzhigaev](https://t.me/abstractDL) & Andrew Kaznacheev

Thanks to [Praxis Relay](https://github.com/josephsteuerjr/praxis-relay) and
[CLIProxyAPI](https://github.com/router-for-me/CLIProxyAPI) for the open-source
ideas and implementations that informed Ouroboros's subscription-backed model
transport through Claudexor.
