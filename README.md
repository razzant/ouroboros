# Ouroboros

<a href="https://github.com/oslook/github-trending/blob/1d61d20a46f66a9590286bf23a8ce8db99be3acf/2026-08-04/python_weekly_trending.json"><img src="assets/github-trending.svg" width="250" height="55" alt="GitHub Trending: #9 Python weekly, August 2026"></a>

[![GitHub stars](https://img.shields.io/github/stars/razzant/ouroboros?style=flat&logo=github)](https://github.com/razzant/ouroboros/stargazers)
[![Downloads](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Frazzant%2Fouroboros%2Fbadges%2Fdownloads.json)](https://ouroboros-agent.ai/install/)
[![Website](https://img.shields.io/badge/website-ouroboros--agent.ai-c93545.svg)](https://ouroboros-agent.ai/)
[![Technical report](https://img.shields.io/badge/technical_report-arXiv%3A2608.08311-b31b1b.svg)](https://arxiv.org/abs/2608.08311)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![macOS 12+](https://img.shields.io/badge/macOS-12%2B-black.svg)][download-macos-arm64]
[![Linux](https://img.shields.io/badge/Linux-x86__64-orange.svg)](https://ouroboros-agent.ai/install/#linux)
[![Windows](https://img.shields.io/badge/Windows-x64-blue.svg)][download-windows-x64]
[![OuroborosHub](https://img.shields.io/badge/OuroborosHub-skills%20marketplace-8A2BE2.svg)](https://github.com/razzant/OuroborosHub)
[![Version 7.0.0](https://img.shields.io/badge/version-7.0.0-green.svg)](VERSION)

Ouroboros is an open-source, general-purpose AI agent whose identity, durable memory, and history continue across tasks and restarts. It works on external projects, coordinates a live swarm of specialist agents, and can rewrite the implementation it runs on, including its code, architecture, prompts, tools, and dependencies. Reflection can also change how it understands itself without severing that continuity.

It runs as a native desktop app or through a headless CLI. The runtime keeps its repository, durable memory, history, and interface on your machine, while model inference can use remote APIs you configure or a local GGUF model.

> **Changing Ouroboros? Coding agents and people must read [CONTRIBUTING.md](CONTRIBUTING.md) before editing.** It defines the required project context, verification, and separate-agent review flow.

## Download Ouroboros

> **Just want to use Ouroboros? Click the download for your platform below. You do not need to clone this repository or install Python or uv.**

- **macOS 12+ on Apple silicon:** [**Download for macOS (.dmg)**][download-macos-arm64]
- **Windows x64:** [**Download for Windows (.zip)**][download-windows-x64]
- **Debian, Ubuntu, or Astra Linux x86_64:** [**Download the Debian package (.deb)**][download-linux-deb-amd64]
- **Fedora or RHEL x86_64:** [**Download the RPM package (.rpm)**][download-linux-rpm-x86_64]
- **RED OS 8 x86_64:** [**Download the RED OS package (.rpm)**][download-linux-rpm-red80-x86_64]
- **Other Linux x86_64:** [**Download the portable AppImage**][download-linux-appimage-x86_64] or the [tar.gz archive][download-linux-x86_64]

Files named `SHA256SUMS`, `release-evidence.json`, `release-smoke-*.json`, and `sbom-*.cdx.json` are verification evidence, not additional installers.

### macOS quick start

1. Click [**Download for macOS (.dmg)**][download-macos-arm64]. The current file is named `Ouroboros-<version>.dmg`.
2. Open the DMG and drag `Ouroboros.app` onto the **Applications** shortcut.
3. Open Ouroboros from Applications. If Gatekeeper asks, right-click the app and choose **Open**.

<p align="center">
  <img src="assets/install-macos.png" width="760" alt="Ouroboros DMG window with a large arrow from Ouroboros.app to the Applications shortcut and Install CLI.command below">
</p>

### Windows quick start

1. Click [**Download for Windows (.zip)**][download-windows-x64].
2. Extract the ZIP.
3. Open the extracted `Ouroboros` folder and run `Ouroboros.exe`.

### Linux quick start

- On Debian, Ubuntu, or Astra Linux, download the `.deb` above and run `sudo apt install ./ouroboros_*_amd64.deb`.
- On Fedora or RHEL, download the generic `.rpm` above and run `sudo dnf install ./ouroboros-*.x86_64.rpm`. RED OS 8 has its own `red80` package.
- On another x86_64 distribution, download the AppImage, make it executable with `chmod +x Ouroboros-*.AppImage`, and run it. Git must already be installed.

To run tasks, configure at least one supported remote provider API key or a local GGUF model. The first-run wizard guides model access, review policy, and budget setup.

<details>
<summary>Optional CLI included with desktop downloads</summary>

The desktop packages already contain an optional CLI installer. On macOS, after copying the app to Applications, double-click `Install CLI.command` in the mounted DMG. On Linux use `./Ouroboros/bin/install-ouroboros-cli`; on Windows use `Ouroboros\bin\install-ouroboros-cli.cmd`. These installers create a user-local `ouroboros` command without sudo. You do not need Python or uv.

</details>

[download-macos-arm64]: https://github.com/razzant/ouroboros/releases/download/v7.0.0/Ouroboros-7.0.0.dmg
[download-windows-x64]: https://github.com/razzant/ouroboros/releases/download/v7.0.0/Ouroboros-7.0.0-windows-x64.zip
[download-linux-deb-amd64]: https://github.com/razzant/ouroboros/releases/download/v7.0.0/ouroboros_7.0.0_amd64.deb
[download-linux-rpm-x86_64]: https://github.com/razzant/ouroboros/releases/download/v7.0.0/ouroboros-7.0.0-1.x86_64.rpm
[download-linux-rpm-red80-x86_64]: https://github.com/razzant/ouroboros/releases/download/v7.0.0/ouroboros-7.0.0-1.red80.x86_64.rpm
[download-linux-appimage-x86_64]: https://github.com/razzant/ouroboros/releases/download/v7.0.0/Ouroboros-7.0.0-linux-x86_64.AppImage
[download-linux-x86_64]: https://github.com/razzant/ouroboros/releases/download/v7.0.0/Ouroboros-7.0.0-linux-x86_64.tar.gz

Ouroboros bundles [Claudexor](https://github.com/razzant/claudexor) as its local execution layer for delegated coding and hosted-agent review. Ouroboros owns the task, memory, review, and final integration, while Claudexor runs the selected connected coding harness and returns durable execution evidence. [Explore Claudexor](https://claudexor.ai/).

The technical report, [Ouroboros: A Self-Developing Frontier Coding Agent with Reviewed Core Evolution](https://arxiv.org/abs/2608.08311), describes the reviewed core-evolution system, the 161-day Hope deployment, and the benchmark campaigns summarized below. [Paper page](https://ouroboros-agent.ai/paper/) · [Hugging Face](https://huggingface.co/papers/2608.08311)

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
| Terminal-Bench 2.1 | Claude Opus-5 high | **86.74%** after zeroing one disclosed reward-hack trial (raw: 86.97%) | Claude Code + Fable 5: 83.8% | Self-reported, submission open | [submission](https://github.com/harbor-framework/terminal-bench-2-1/pull/175) · [run](https://hub.harborframework.com/jobs/2b145543-edeb-4a3b-b46f-4800310f1182) |
| Terminal-Bench 2.1 | Claude Opus-4.8 high | **80.22%** | Claude Code: 78.9% | Self-reported, public run | [run](https://hub.harborframework.com/jobs/4b8e244f-8ab0-4d28-8218-7cf346282faa) |
| Terminal-Bench 2.1 | GPT-5.5 | **84.3%** | Codex CLI: 83.1% | Self-reported, public run | [run](https://hub.harborframework.com/jobs/f02fd019-23e1-495f-af0a-ebd9a65f3079) |
| Terminal-Bench 2.1 | Grok-4.5 | **84.94%** after a reward-hack audit | Cursor CLI: 79.3% · Hermes: 77.53% | Self-reported, submission open | [submission](https://github.com/harbor-framework/terminal-bench-2-1/pull/146) |
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

Use your existing **Codex, Claude Code, or Cursor subscriptions** for delegated coding and review. Ouroboros drives them through [Claudexor](https://github.com/razzant/claudexor), its bundled multi-harness engine. Connect accounts in **Settings → Agents**; no separate Claudexor install is needed. Release artifacts carry the exact reviewed engine and Node archives. Source checkouts obtain those same pinned archives on first use.

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

External workspaces must be separate Git worktree roots and may not overlap Ouroboros's own repository or data directory. Patch, streaming, detached-task, and schedule semantics are documented in the CLI help and the canonical [architecture](docs/ARCHITECTURE.md).

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

To change Ouroboros itself, follow [CONTRIBUTING.md](CONTRIBUTING.md) and read [BIBLE.md](BIBLE.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md), and [docs/CHECKLISTS.md](docs/CHECKLISTS.md) in full before editing.

#### Configuration

The first-run wizard and **Settings** configure model access, cognitive roles, local models, review policy, runtime mode, budget, skills, and optional integrations. Ouroboros supports configurable remote providers, compatible endpoints, and local GGUF inference; exact settings and defaults live in [`ouroboros/config.py`](ouroboros/config.py) and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

The server binds to `127.0.0.1:8765` by default. Read [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) before exposing it beyond loopback; non-local binds need `OUROBOROS_NETWORK_PASSWORD` or an explicitly trusted external access layer.

#### Run Tests

```bash
make test
```

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

Output: `dist/Ouroboros-<VERSION>.dmg`, containing `Ouroboros.app`, an `Applications` shortcut, and `Install CLI.command`. Omit `OUROBOROS_SIGN=0` when a Developer ID signing identity is configured.

### Linux (.AppImage and .tar.gz)

```bash
bash scripts/download_python_standalone.sh
bash build_linux.sh
```

Outputs: `dist/Ouroboros-<VERSION>-linux-<arch>.AppImage` and the extraction-friendly `dist/Ouroboros-<VERSION>-linux-<arch>.tar.gz`. The AppImage needs host Git; run it after `chmod +x`, or pass `--cli` to reach its bundled CLI. If FUSE is unavailable, set `APPIMAGE_EXTRACT_AND_RUN=1` when launching it. The tarball contains `./Ouroboros/bin/install-ouroboros-cli`. If bundled browser tools need host libraries, run `./Ouroboros/_internal/python-standalone/bin/python3 -m playwright install-deps chromium webkit` from the extracted tarball.

On a build host where system packages are managed separately, set `OUROBOROS_SKIP_PLAYWRIGHT_INSTALL_DEPS=1`; Chromium and WebKit are still downloaded and bundled, but the build does not invoke `sudo` to install host libraries.

### Linux (.deb and .rpm)

Wraps the payload `build_linux.sh` just produced, so run it afterwards:

```bash
sudo apt-get install -y dpkg-dev rpm   # rpm provides rpmbuild
bash scripts/build_linux_packages.sh
```

Output: `dist/ouroboros_<VERSION>_amd64.deb`, `dist/ouroboros-<VERSION>-1.x86_64.rpm` and `dist/ouroboros-<VERSION>-1.red80.x86_64.rpm` (RED OS 8). All three declare Git as a runtime dependency and install to `/opt/ouroboros` with a `/usr/bin/ouroboros` symlink, a desktop entry, and an opt-in systemd user unit. The Linux launcher is built by the bundled portable Python so the build runner cannot raise its glibc floor. `bash scripts/smoke_linux_packages.sh official <deb> <rpm> <red80-rpm>` installs all three through `apt` or `dnf` in Ubuntu 22.04 and Fedora 42 containers, resolves Git, verifies the installed unit, and checks both the real CLI and a bounded desktop-launcher start; this lane gates the release. Swap `official` for `vendor` to repeat the check on Astra Linux 1.8 and RED OS 8 images from the vendors' own registries — that lane runs informationally in CI, so an outage at a third-party registry cannot block a tagged release.

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
for the complete agent-first workflow. Open pull requests against lowercase
`ouroboros`, leave release-version allocation to maintainers, and have a
separate agent context review the final diff. Any coding harness or configured
review route may produce the evidence; if none is available, record `NOT_RUN`
and the reason.

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| 7.0.0 | 2026-08-20 | **refactor: v7 — the whole runtime moves to bounded domain leaves behind permanent facades.** The largest structural release in the project's history, executed as a reviewed refactoring campaign: 74 over-band modules (largest 7,314 lines) become 411 runtime modules across 19 domains, every module within the 1,500-line band and the size-debt manifest EMPTY (`GIANT_PATHS`, `MODULE_DEBT_1500`, `BYTE_DEBT` all cleared); the seven test giants split into 31 themed suites (435 → 664 test files). Semantics are byte-preserved except a ledgered, owner-approved delta list: every relocation is a row in `MIGRATION_v7.md`, machine-validated by the ledger inventory suite, and the module-handle call-time binding idiom (the D18/D33 family) keeps every facade monkeypatch-transparent. External review of contributor PRs now ALWAYS executes on the trusted base's review machinery (D31), retiring the mutable import-classifier boundary. New reader artifacts ship in-tree: `docs/DOMAIN_MAP.md` (19 domains, 1:1 module ownership), `docs/PERSISTENCE_OWNERS.md` (every durable path's writers, readers and lifecycle), `docs/FACADE_CONSUMERS.md` and `docs/TEST_DISPOSITION.md`. Runtime behavior, the frozen Tool API and the gateway contract are unchanged outside the approved delta list. |
| 6.105.1 | 2026-08-18 | **chore: the managed Claudexor runtime pin advances to 3.6.0 — the unified account model activates for every install.** The reviewed pin (`ouroboros/claudexor_runtime_pin.json`) moves 3.5.0 → 3.6.0, the engine release whose headline is the unified account model: every account becomes a named registry row, and the pool-authority read `GET /v2/account-pools` is the feature marker the accounts surface detects. On first daemon start the engine migration turns existing default-store logins into named removable rows, so the v6.105.0 Accounts UI (honest per-profile rows, Enabled toggle, Remove, next-up badge) lights up on every install without manual steps. Pin fields move together — version, build sha, archive URL, sha256, size; protocol major 3 and the Node artifact set (24.16.0) are unchanged. A cross-repo byte-assertion now pins the feature-detect operation id (`get:account-pools`) from both repositories, so a respelling on either side goes red instead of silently degrading installs to the legacy accounts rendering. |
| 6.105.0 | 2026-08-18 | **feat: unified accounts, delegation substrate, rotation visibility.** The first tagged release since v6.103.0 — it also carries the untagged 6.104.0 below. The unified-accounts sprint lands a dual-engine account model behind feature detection (an unreadable engine catalog fails closed to the previous behavior): the Accounts UI renders every engine-side profile with honest copy (the "Default CLI login" / "Managed by the X CLI" fictions are retired), enabled+signed-in family counting with a distinct all-disabled state, an Enabled toggle riding a new PATCH credential-profile thin proxy, Remove on every engine-supported row, and a next-up badge; delegation learns an explicit account pin (`OUROBOROS_SUBAGENT_PROFILE` rides the stored canonical body as `credentialProfileId`, replayed byte-identically on retry, with strict per-subject health for every pinned lane including review-session recovery, and requested-vs-applied custody disclosed on the Last delegated run). The rotation-visibility sprint makes lane state typed and visible end to end: `{failure_code, reset_at, http_status}` plus `capability_delta` travel from substrate to render, review panels count paid only when dispatched and DEGRADED replays honestly, pre-fanout health-skip rows carry a material health epoch and reviewer-roster fingerprint as replay identity, `quorum_unreachable` lands as an honest `blocked_with_evidence` terminal, `schedule_followup` gives agents a one-shot deferred wake through the existing scheduler, and quota rotation becomes a GET→conditional-POST reconcile with a durable receipt that never overwrites an explicitly persisted value. The delegation-substrate work closes a claudexord admission race, restates the nanny's delegation mandate, and passes a pinned credential profile through `route_health` to the engine. |
| 6.104.0 | 2026-08-17 | **feat: the Antigravity (agy) subscription is picked up from Claudexor 3.5.0.** The managed engine pin advances to 3.5.0 (build `efff2f3b`; protocol 3 and Node 24.16.0 unchanged; the archive verified against the release's `runtime-manifest.json` AND `SHA256SUMS`, identity probe green), whose headline is Google's Antigravity CLI as a fourth harness with multi-account quota rotation. The install-preset compiler now RECOGNIZES agy — `HARNESS_AGY` joins `PRESET_HARNESSES`, effort rides inside the model slug like cursor, and an alias table covers the gemini families (`gemini-3.1-pro` publishes high/low only) — while the ratified matrix deliberately stays at the seven claude/codex/cursor combinations: a connected combination without a matrix row compiles to a typed `matrix_row_absent` refusal, checked before discovery validation, replacing a bare `KeyError` that would have turned onboarding completion into an unhandled 500 (the agy seats are an owner decision, dictated separately after living with Antigravity in reviewer roles). The Accounts connect flow learns the no-default-store shape: a first-account login the engine refuses at create time (HTTP 400 with no detected account in the family — no named profile and no detected native login) switches the login card into a name-this-account state carrying the engine's own message instead of a dead-end error — structural, never keyed to a harness name. |
| 6.103.0 | 2026-08-16 | **feat: plan review becomes a domain-neutral spec gate — one review organ pointed at an INTENTION.** `plan_task` now reviews a typed spec (goal, in_scope, non_goals, acceptance_claims, invariants, decisions with rejected alternatives, deferred, affected_resources, evidence) with host-minted ids as the only valid `breaks` targets; reviewers return typed findings (blocking with a `breaks` id / note / need_evidence) and the HOST computes the aggregate — no reviewer authors a competing plan or emits GREEN as authority. ONE structural fact tiers the governance pack: `constitutional` iff a declared target resolves under the system repo (BIBLE attached; everyone else gets a runtime BIBLE navigation map). Agent-declared evidence is attached bounded with every absence named, the runtime data plane denied and content secrets redacted; agent sessions see the same redacted snapshot. New shared owner setting `OUROBOROS_REVIEW_MAX_CYCLES` (default 2, `unlimited`) bounds paid cycles for plan review, task acceptance (passes = cycles − 1; the retired acceptance-passes key migrates at load) and the commit gate's identical-diff attempt cap (default 3 → 2). An identical envelope replays for free; a fully-rejected REVISE_PLAN wave earns exactly one paid delta cycle; under blocking an open plan holds implementation and a spent cap lands a typed `review_cycles_exhausted` escalation with an honest `blocked_with_evidence` terminal. Deleted: planning scouts, the plan Atlas, `plan_class`, `context_level`, the governance mega-pack, the generative reviewer stance, the hidden 32-wave limit and the api_chat-only pin. BIBLE P3 gains one sentence separating a reviewer of a CHANGE from a reviewer of an INTENTION; the Plan Review Checklist is rewritten findings-only around one question. Net −3.6k lines. Live measurement: a self-modification plan review costs ~$0.14 per wave against ~$3.5 before. |
| 6.102.0 | 2026-08-15 | **feat: the owner gets real mid-task controls, and the context engine is redesigned — a five-stream synthesis.** "Wrap up" turns owner cancellation into finalization with custody: the worker gets one bounded finalize episode to synthesize an honest LLM-written summary of what was actually done, anchored to when the control is DELIVERED to the model (not when the button is pressed, so a long blocking tool call no longer eats the window), with an outer 10-minute honesty cap from the owner's request, an immediate-custody path for `grace_sec<=0`, a zero-paid-turn forced fallback when the control arrives already expired, and a durable drain stamp whose persistence failure is retried once and then disclosed as a typed forensic event instead of silently widening the budget; every terminal seam (expiry, handoff, salvage) lands as a typed receipt rendered in the UI, and the summary is preserved — never clobbered — by the salvage path. Beside it, the "Hurry up" control asks a running task to converge without killing it, delivered through the owner mailbox invisibly to chat history. Context handling is redesigned around a typed ContextFit deficit/reclaim contract instead of ad-hoc compaction thresholds. Skill payloads become editable through delegated harness sessions with snapshot custody. The web UI gains delegation and review cards, harness login cards, and a task-control dropdown — all user-facing strings English per the product language policy. Delegation-first economics are reasserted: the system prompt now frames delegation as the default for ANY substantial work (research and documents included, `web_search` reserved for point lookups on harness children), the first nanny reminder fires at round 3, nanny notes ride typed checkpoints instead of spamming owner chat as progress warnings, and a planning swarm whose scout is cancelled exits as scout-unavailable instead of running a paid reviewer panel over a dead wave. Claudexor login custody is fixed so a login-owned daemon survives its session. |
| 6.101.1 | 2026-08-13 | **fix: Windows CI compatibility.** Packaging and delegated-run test coverage now avoids platform-specific assumptions while preserving their intended behavior. |
| 6.101.0 | 2026-08-13 | **refactor: exact-path repository size ratchets.** Module and function exceptions now live in one checked-in manifest keyed by repository-relative path and qualified symbol, with deterministic regeneration and shrink-only validation. The same policy feeds smoke tests, hermetic preflight, health reporting, protected-path resolution, update policy, and CI while preserving the existing ceiling, runtime behavior, and public contracts. |
| 6.97.2 | 2026-08-11 | **fix: nested AppImage cleanup has a real lifecycle owner, and path guards keep the same fail-closed meaning on every supported Python.** The marker-gated `AppRun` now remains between the type-2 runtime and desktop launcher, waits for the launcher recorded by the PID file, removes only its verified extraction, and removes the empty private runtime base before returning the payload status. Linux release smoke proves that process chain and both cleanup boundaries. A shared allow-missing resolver first validates existing path ancestry, so Python 3.13's changed symlink-loop behavior can no longer turn an unresolvable delegated write root, read target, or harness home into a partially resolved path; ordinary missing targets remain supported. Packaging and delegated-containment correctness only; workspace authority and frozen Tool API contracts are unchanged. |
| 6.97.1 | 2026-08-11 | **fix: Linux AppImage relaunches use independent extraction roots, and Windows release builds avoid a false dirty-tree failure.** Nested extract-and-run starts no longer share the outer CLI runtime's temporary payload; Linux release smoke waits for the owning runtime to finish cleanup instead of racing a fixed pathname timeout. The generated runtime lock is pinned to LF so `uv export` preserves the strict clean-tree bundle gate on Windows, and that gate now reports offending paths. Packaging and release-CI only; workspace authority and frozen Tool API contracts are unchanged. |
| 6.96.2 | 2026-08-11 | **fix: the hermetic preflight capture is byte-faithful on Windows too, and a POSIX-only filename test is skipped there.** The v6.96.1 release full-test matrix (first tag build since the byte-faithful capture landed) caught two windows-latest failures in the preflight runner. The hermetic worktree checked HEAD out and applied the `--binary` candidate diff under the runner's default `core.autocrlf=true`, so an LF payload landed on a CRLF-converted base and every line ending was mangled — breaking the exact byte-faithfulness the capture exists to hold; the `worktree add` and `git apply` now pin `core.autocrlf=false` + `core.eol=lf` (no `.gitattributes text` directive governs the affected paths, so the override is authoritative and inert on POSIX). The second failure was a synthetic non-UTF-8 filename test asserting an os.fsdecode round-trip that only holds under POSIX surrogateescape: Windows uses a UTF-16 filesystem where such a name cannot exist and its fs codec raises on the injected 0xff byte, and git never emits such a name there, so the production path is unaffected — the test is now skipped on Windows with that reason. Test/CI-correctness only; no runtime behavior change on any platform. |
Older releases are preserved in Git tags and GitHub releases. Older 6.x rows (including 6.97.0, 6.96.1, 6.96.0, 6.95.0, 6.94.0, 6.93.0, 6.92.1, 6.92.0, 6.91.1, 6.90.3, 6.91.0, 6.90.2, 6.90.0, 6.87.5, 6.87.4, 6.87.3, 6.87.2, 6.84.0, 6.87.1, 6.83.0, 6.86.1, 6.81.1, 6.76.0, 6.75.0, 6.74.5, 6.74.4, 6.74.1, 6.74.0, 6.73.2, 6.73.1, 6.73.0, 6.72.0, 6.71.2, 6.71.1, 6.71.0, 6.70.0, 6.69.0, 6.68.0, 6.67.0, 6.66.0, 6.65.4, 6.65.3, 6.65.2, 6.65.1, 6.65.0, 6.64.3, 6.64.2, 6.64.1, 6.64.0, 6.63.0, 6.62.0, 6.61.4, 6.61.3, 6.61.1, 6.61.0, 6.60.0, 6.59.0, 6.58.0, 6.57.0, 6.56.0, 6.55.0, 6.54.4, 6.54.2, 6.54.1, 6.54.0, 6.53.4, 6.53.0, 6.51.0), the 5.2.0 through 5.33.0-rc.6 rows, and former `4.0.0` rows are rolled off to respect the P9 changelog cap; their full bodies remain at their git tags.

---

## License

[MIT License](LICENSE)

Created by [Anton Razzhigaev](https://t.me/abstractDL) & Andrew Kaznacheev
