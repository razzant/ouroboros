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
[![Version 7.5.0](https://img.shields.io/badge/version-7.5.0-green.svg)](VERSION)

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

**Android (experimental, Magisk-rooted ARM64):** [Installation and qualification guide](docs/ANDROID_INSTALL.md). Its USB setup requires Python and adb.

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

[download-macos-arm64]: https://github.com/razzant/ouroboros/releases/download/v7.5.0/Ouroboros-7.5.0.dmg
[download-windows-x64]: https://github.com/razzant/ouroboros/releases/download/v7.5.0/Ouroboros-7.5.0-windows-x64.zip
[download-linux-deb-amd64]: https://github.com/razzant/ouroboros/releases/download/v7.5.0/ouroboros_7.5.0_amd64.deb
[download-linux-rpm-x86_64]: https://github.com/razzant/ouroboros/releases/download/v7.5.0/ouroboros-7.5.0-1.x86_64.rpm
[download-linux-rpm-red80-x86_64]: https://github.com/razzant/ouroboros/releases/download/v7.5.0/ouroboros-7.5.0-1.red80.x86_64.rpm
[download-linux-appimage-x86_64]: https://github.com/razzant/ouroboros/releases/download/v7.5.0/Ouroboros-7.5.0-linux-x86_64.AppImage
[download-linux-x86_64]: https://github.com/razzant/ouroboros/releases/download/v7.5.0/Ouroboros-7.5.0-linux-x86_64.tar.gz

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

The server binds to `127.0.0.1:8765` by default. Read [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) before exposing it beyond loopback; non-local binds need `OUROBOROS_NETWORK_PASSWORD` or an explicitly trusted external access layer.

#### Run Tests

```bash
python scripts/run_tests.py                  # full local battery (same as `make test`)
python scripts/run_tests.py tests/test_x.py  # focused run
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

Docker runs the web runtime, not the native desktop shell. The image bundles Chromium and WebKit for the locked
Playwright version and needs Docker's default BuildKit builder; use [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for network
and container policy, including how to trust an extra CA (`OUROBOROS_EXTRA_CA_BUNDLE`).

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
| 7.5.0 | 2026-09-26 | **feat: one self-contained Docker image and an owner extra-CA bundle for provider TLS (#1300).** Contributed by @komsikov and reworked in-branch: the image installs the locked Playwright browsers above the lock copy with BuildKit cache mounts (4.82 GB → 4.12 GB; the tag-only docker lanes exercise the shipped browsers), and `OUROBOROS_EXTRA_CA_BUNDLE` adds a PEM of extra CA certificates on top of certifi for every first-party provider call, with a content-addressed merged bundle so a rotated certificate takes effect on the next task (Settings → Advanced; `docs/DEPLOYMENT.md`). |
| 7.4.11 | 2026-09-24 | **chore: ignore `graphify-out` and add a root `.dockerignore` (#1255).** Contributed by @komsikov; the Docker build context keeps secrets, host environments and local runtime/review state out of image layers while `.git`, `tests/` and sources stay in, pinned by `TestDockerignore`. |
| 7.4.10 | 2026-09-24 | **fix: isolate candidate verification and run complete event-driven browser CI (#1215, #1143, #1084, #1112).** Tests, preflight and server children run on disposable HOME/data/projects/worktrees/Deliverables roots and never install dependencies into the operator's Python; a browser candidate is a faithful copy of the dirty tree (staged, unstaged, new, deleted, binary, mode) whose source HEAD, index and files are proven unchanged, with unsupported inputs and concurrent mutation refused explicitly and temporary trees retained until process teardown is proven. The complete existing `ui_browser` lane runs for every non-documentation PR and every `ouroboros` push through `--require-ui-browser` (no new cron); the startup history-audit child inherits the containment token; the isolation keeps the platform's default text encoding so Windows CI still catches cp1252 bugs. |
| 7.4.9 | 2026-09-24 | **fix: truthful task cards and history (#1011, #1087, #1110, #1154, #1061, #931, #1073, #516, #869, #498).** A known outcome (Failed / Done with warnings) is the card's primary word while post-task work still runs, with `Finalizing…` secondary and the task name stable; a completed lifecycle with a failed outcome never reads as a clean completion. Money speaks one scoped carrier — `Cost unknown`, `Tracked: $X`, `up to $X`, an evidenced `$0.00` — with `cost unavailable` reserved for an unreadable ledger and a nested child's foreign rollup never replaced by its own zero. Cancellation states its recorded cause and proven parent relation; task diagnostics stay in the disclosure, worker events in Logs; the Project's final Main summary is durably owed, retried and deduplicated after completion, and the provider-outage terminal names its source and a human cause without extra model calls. Verified by focused, Node and real-backend browser journeys including a live Failed+Finalizing actor. |
| 7.4.8 | 2026-09-24 | **chore: ignore `.devcontainer` (#1253).** A contributor's devcontainer folder no longer dirties the tree or the seed gate. |
| 7.4.7 | 2026-09-24 | **test: repair Windows repository-evidence fixtures.** Keep POSIX private-mode checks where supported and clear the read-only attribute before deliberately removing a fixture Git object; byte-retention and unavailable-source checks remain active on Windows. |
| 7.4.0 | 2026-09-20 | **feat: add an honest desktop attention cue for live notifications.** The optional launcher bridge can raise the existing desktop window and request one platform system sound, returning explicit native/unsupported/window_only/unavailable facts; it does not claim Notification Center delivery, run after close, or add a tray/background process. Browser banners and in-app toasts remain the fallback, with capability text visible in Settings. Focused native, launcher and notification tests cover the seam. |
| 7.3.0 | 2026-09-19 | **feat: notifications can pull you back to a question or a finished task while a client is running (#1122).** Settings → Appearance gains a notification block next to the theme: a master switch, the two required categories (a question or decision is waiting for you, a task finished or stopped), a model-chosen category for messages Ouroboros sends while it works, a separate off-by-default toggle for ordinary replies in Main, sound, an off-by-default show-the-text choice and a test notification. The choices are stored per client exactly like the appearance choice, never reach the server, and do not make the settings draft dirty. Delivery is page-level: a system banner where this client exposes one and permission is granted, otherwise the in-app surface plus one short tone, with the status line saying which surface you actually have; either way a click opens the source. The new `web/modules/notifications.js` keeps classification and the delivery gate pure over one live frame plus the stored preferences, and takes ONE subscription per client on the shared socket — not inside a chat instance, which dies with its room, so a Project the owner never opened still reaches him; `web/modules/chat.js` is untouched. Only live frames reach it, so a reload cannot re-notify without any stored notification state. A finished managed root is recognised on the shape it actually arrives in (a live `task_done` log frame) as well as the authored summary, and a conversation turn's own ending stays the ordinary-reply category instead of claiming a task finished; all of them collapse to one key per task. A child task never notifies the owner: it escalates to its parent, and lineage is read from the delegation facts frames carry because the terminal frame has none. Importance needs no new host field and no second model call: the existing proactive-message discriminator is the signal. Policy and its disclosed limits have one canonical home in `docs/DESIGN.md` §9 — including that each open window is its own client, that an event during a dropped socket never rings, and that no OS permission, Do Not Disturb or platform limit is bypassed. |
| 7.2.0 | 2026-09-19 | **feat: Light, Dark and System appearance, review that reads the repository instead of a packed snapshot, and one fast local test battery.** The web UI gains a client-local Appearance choice — Light, Dark or System — applied before first paint in the app and the onboarding wizard and carried through in-page charts, diagrams and host-rendered widget charts, while framed skill widgets keep their own colours (#1107). With no saved choice the UI follows the system scheme, so an existing install on a light OS opens in Light until Dark is chosen; to remember the choice the desktop shell now keeps website data, including cookies, and a desktop app that only took in-app updates remembers it once the 7.2.0 installer has been installed. Project questions are one row in Main and become a card only while the task waits (#1106). Scope review retrieves what it needs from the repository rather than receiving one packed snapshot (#1043); plan review keeps paid feedback (#1051), acceptance review keeps its rework controls, final review text and explicit author completion (#1042, #1069, #1089), and system review mail no longer reopens accepted results (#1088). `python scripts/run_tests.py` is the one documented local battery: it runs the node lane first, then every default-lane test in one parallel run (#1101). First-run setup recovers subscription sign-in and model discovery (#1099), and Finish stays usable after a refused Main-reviewer recovery (#1100); delegated requests are stored by reference and keep pending recovery (#1067); terminal outcomes, usage accounting, cancellation provenance, task relay authorship and process environments are preserved end to end (#1054, #1064, #1083, #1098). Also: the managed Claudexor runtime moves to 3.12.4, an experimental Android host with optional release artifacts (#873), bounded Cowork Bench meter reads that preserve interrupted work (#1044), chat composer, Activity row and project-link fixes (#1063, #1095), clearer Presence delivery (#1053), subagent scheduling refusals (#1092), settings and helper-history reporting (#1090), and scope-review context logging from a community contribution (#752). |
| 7.1.0 | 2026-09-17 | **feat: memory that understands people, Background Consciousness as an ordinary Main turn, and a chat that leads with narration.** Memory keeps an understanding of people in front of the mind through resident summaries, one memory shared by every room and honest consolidation (#910). Background Consciousness becomes an ordinary Main turn on an alarm clock (#988, closing the class behind #654). Chat blocks lead with narration while tool calls fold into one evidence row (#1007); task cards follow their work and treat acceptance review as advice to Ouroboros (#970); question pointers say whether an answer is wanted and show the recorded answer (#1025); the unified UI control system, source picker and model chooser name models and accounts truthfully (#768, #862, #890). Planning is asynchronous and Swarm tasks are admitted directly (#851); delegated work survives durable cancellation (#850); a task's own reviewer is never swept as its delegation (#1008); semantic duplicate vetoes leave subagent admission (#887); Presence profiles work in an owner-selected folder (#941). Prompt caches extend through an append-only acceptance observation and a shared Codex cache key per install and model (#929, #1015). Update letters include merged branch changes (#1003), the owned Claudexor daemon latches failed starts with a typed diagnosis, and the managed runtime advances to Claudexor 3.12.1 (#891, #962), alongside Windows, platform-installation and UI-smoke repairs. |
| 7.0.0 | 2026-09-08 | **v7: a modular runtime with full ordinary-conversation tools, complete delegated inputs and visible owner dialogue.** Main and Project conversations retain their chosen tools and working folders; required owner waits preserve the live browser without occupying pooled execution capacity. Publish admission is visible in its intended chat, and GitHub/image/plan failures retain their actual causes. Planning advice stays optional; source-backed evidence, separate host notices and exact-workload test reuse preserve honest review without new approval machinery. Integrates the subscription model and account-role controls, owned startup/restart custody and cross-platform repairs, with Claudexor pinned to 3.10.1. |
Older releases are preserved in this repository's history. Rows 7.4.5, 6.114.0, 6.113.2 and 6.113.5 are retained in Git history. Older 6.x rows (including 6.113.1, 6.110.0, 6.109.0, 6.110.1, 6.108.1, 6.106.0, 6.101.1, 6.97.2, 6.105.0, 6.97.1, 6.97.0, 6.96.1, 6.96.0, 6.95.0, 6.94.0, 6.93.0, 6.92.1, 6.92.0, 6.91.1, 6.90.3, 6.91.0, 6.90.2, 6.90.0, 6.87.5, 6.87.4, 6.87.3, 6.87.2, 6.84.0, 6.87.1, 6.83.0, 6.86.1, 6.81.1, 6.76.0, 6.75.0, 6.74.5, 6.74.4, 6.74.1, 6.74.0, 6.73.2, 6.73.1, 6.73.0, 6.72.0, 6.71.2, 6.71.1, 6.71.0, 6.70.0, 6.69.0, 6.68.0, 6.67.0, 6.66.0, 6.65.4, 6.65.3, 6.65.2, 6.65.1, 6.65.0, 6.64.3, 6.64.2, 6.64.1, 6.64.0, 6.63.0, 6.62.0, 6.61.4, 6.61.3, 6.61.1, 6.61.0, 6.60.0, 6.59.0, 6.58.0, 6.57.0, 6.56.0, 6.55.0, 6.54.4, 6.54.2, 6.54.1, 6.54.0, 6.53.4, 6.53.0, 6.51.0), the 5.2.0 through 5.33.0-rc.6 rows, and former `4.0.0` rows are rolled off to respect the P9 changelog cap; their full bodies remain in this file's git history, with release tags where present for historical versions.

---

## License

[MIT License](LICENSE)

Created by [Anton Razzhigaev](https://t.me/abstractDL) & Andrew Kaznacheev

Thanks to [Praxis Relay](https://github.com/josephsteuerjr/praxis-relay) and
[CLIProxyAPI](https://github.com/router-for-me/CLIProxyAPI) for the open-source
ideas and implementations that informed Ouroboros's subscription-backed model
transport through Claudexor.
