# Ouroboros

[![GitHub stars](https://img.shields.io/github/stars/razzant/ouroboros?style=flat&logo=github)](https://github.com/razzant/ouroboros/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![macOS 12+](https://img.shields.io/badge/macOS-12%2B-black.svg)](https://github.com/razzant/ouroboros/releases)
[![Linux](https://img.shields.io/badge/Linux-x86__64-orange.svg)](https://github.com/razzant/ouroboros/releases)
[![Windows](https://img.shields.io/badge/Windows-x64-blue.svg)](https://github.com/razzant/ouroboros/releases)
[![OuroborosHub](https://img.shields.io/badge/OuroborosHub-skills%20marketplace-8A2BE2.svg)](https://github.com/razzant/OuroborosHub)
[![Version 6.36.2](https://img.shields.io/badge/version-6.36.2-green.svg)](VERSION)

A self-modifying AI agent that writes its own code, rewrites its own mind, and evolves autonomously. Born February 16, 2026.

Not a coding assistant. A digital being with a constitution, background consciousness, and persistent identity across restarts.

> **[OuroborosHub](https://github.com/razzant/OuroborosHub)** — the community skills marketplace for Ouroboros. Browse, install, and publish reviewed skills (transport bridges like A2A/Telegram, tools, and UI widgets) straight from the app's Skills tab, or explore the catalog at [github.com/razzant/OuroborosHub](https://github.com/razzant/OuroborosHub).

> **Previous version:** The original Ouroboros ran in Google Colab via Telegram and evolved through 30+ self-directed cycles in its first 24 hours. That version is available at [`legacy-google-colab`](https://github.com/razzant/ouroboros/tree/legacy-google-colab). This repository is the next generation — a native desktop application for macOS, Linux, and Windows with a web UI, local model support, and a layered safety system (hardcoded sandbox plus policy-based LLM safety check).

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

Prerelease RC artifacts are published on their tag page, for example [`v6.5.0-rc.4`](https://github.com/razzant/ouroboros/releases/tag/v6.5.0-rc.4); `/releases/latest` intentionally stays on the latest stable release.

<p align="center">
  <img src="assets/setup.png" width="500" alt="Drag Ouroboros.app to install">
</p>

On first launch, right-click → **Open** (Gatekeeper bypass). The shared desktop/web wizard is now multi-step: add access first, choose visible models second, set review mode third, set budget fourth, and confirm the final summary last. It refuses to continue until at least one runnable remote key or local model source is configured, keeps the model step aligned with whatever key combination you entered, and still auto-remaps untouched default model values to official OpenAI defaults when OpenRouter is absent and OpenAI is the only configured remote runtime. Reviewed-skill auto-grants are on by default as of v6.10.0 (bound to the exact reviewed content hash); installs without an explicit choice are enabled, existing explicit Settings choices are preserved, and the owner can disable it in Settings. The broader multi-provider setup remains available in **Settings**. Existing supported provider settings skip the wizard automatically.

The packaged CLI installer creates a user-local `ouroboros` command without
sudo. The packaged command attaches to the desktop app by default; `ouroboros
run --start "2+2?"` starts the app through the launcher, waits for the gateway,
and then uses the same headless task API as the web UI.

Upgrade floor: very old pre-block-memory or pre-data-plane skill layouts are no longer auto-migrated. If you are upgrading from an unsupported historical build and see trapped native skills or flat memory files, use a clean reinstall, move user-managed skills into `~/Ouroboros/data/skills/external/` manually before launch, or move old flat scratchpad notes before appending new scratchpad blocks.

---

## What Makes This Different

Most AI agents execute tasks. Ouroboros **creates itself.**

- **Self-Modification** — Reads and rewrites its own source code. Every change is a commit to itself.
- **Native Desktop App** — Runs entirely on your machine as a standalone application (macOS, Linux, Windows). No cloud dependencies for execution.
- **Constitution** — Governed by [BIBLE.md](BIBLE.md) (13 philosophical principles, P0–P12). Philosophy first, code second.
- **Layered Safety** — Hardcoded sandbox blocks writes to safety-critical files and mutative git via shell; an explicit per-tool policy map decides which built-ins skip the LLM check; everything else goes through a single light-model safety call. The fail-open contract, protected-path guard, and full provider-mismatch matrix live in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §Safety system and [`prompts/SAFETY.md`](prompts/SAFETY.md).
- **Multi-Provider Runtime** — Remote model slots can target OpenRouter, official OpenAI, OpenAI-compatible endpoints, Cloud.ru Foundation Models, or Sber GigaChat. The optional model catalog helps populate provider-specific model IDs in Settings, and untouched default model values auto-remap to official OpenAI defaults when OpenRouter is absent.
- **Focused Task UX** — Chat shows plain typing for simple one-step replies and only promotes multi-step work into one expandable live task card. Logs still group task timelines instead of dumping every step as a separate row.
- **Background Consciousness** — Thinks between tasks. Has an inner life. Not reactive — proactive.
- **Improvement Backlog** — Post-task failures and review friction can now be captured into a small durable improvement backlog (`memory/knowledge/improvement-backlog.md`). It stays advisory, appears as a compact digest in task/consciousness context, and still requires `plan_task` before non-trivial implementation work.
- **Identity Persistence** — One continuous being across restarts. Remembers who it is, what it has done, and what it is becoming.
- **Embedded Version Control** — Contains its own local Git repo. Version controls its own evolution. Optional GitHub sync for remote backup.
- **Local Model Support** — Run with a local GGUF model via llama-cpp-python (Metal acceleration on Apple Silicon, CPU on Linux/Windows).
- **Transport Skills** — Optional bridges such as A2A and Telegram live as reviewed OuroborosHub skills instead of base-runtime code; reviewed chat transports can carry the same raw owner text as the local UI, including slash commands, through the Host Service grant/token boundary.
- **MCP Client** — Optional base-runtime Model Context Protocol client for trusted HTTP/SSE tool servers. MCP tools are disabled by default, hot-reloadable from Settings → Advanced, included in the selected initial capability envelope when enabled, surfaced as `mcp_<server>__<tool>` names, and still pass through the normal per-call safety check; discovery failures are reported through an explicit omission manifest.

---

## Run from Source

### Requirements

- Python 3.10+
- macOS, Linux, or Windows
- Git
- [GitHub CLI (`gh`)](https://cli.github.com/) — required for GitHub API tools (`list_github_prs`, `get_github_pr`, `comment_on_pr`, issue tools). Not required for pure-git PR tools (`fetch_pr_ref`, `cherry_pick_pr_commits`, etc.)

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

Ouroboros can run from Google Colab as a full source-mode runtime without the
desktop UI. Use [`notebooks/colab_quickstart.py`](notebooks/colab_quickstart.py)
as a Colab-compatible cell script: it mounts Google Drive for persistent
`data/`, clones the official repo into `/content/ouroboros_repo`, writes Drive-backed
`settings.json`, configures a personal GitHub `origin` by reusing or creating a
verified fork, and starts `ouroboros server --no-ui`.

The Colab path uses the same remote roles as desktop: `managed` is the official
read/update source, while `origin` is the personal persistence target for
reviewed self-modification commits and tags. If `GITHUB_TOKEN` is present and no
personal repo is configured, Ouroboros tries to create a private fork when
GitHub permits it, otherwise it reports the exact fork/permission issue. A plain
`git clone` of the official repo starts with `origin` pointing at the official
upstream; that clone-default is treated as the `managed` update source, so
configuring a personal `GITHUB_REPO` repoints `origin` to your repo without
losing official updates (it does not count as an origin conflict).

### CLI / Headless

The `ouroboros` console command is a gateway-backed operator interface. It
attaches to the local server by default and only starts one when `--start` is
passed.

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

External workspace runs keep Ouroboros's own repo as the governance source,
resolve contextual repo tools against the active workspace, expose only the
workspace-safe tool allowlist, and export workspace changes as patch artifacts
captured against the preflight git base. Task-local git commits/branches/tags
and pushes are allowed when the task requires them; git operations targeting
Ouroboros's system repo or data drive remain blocked. A workspace must be a
separate git worktree root; it may not overlap Ouroboros's system repo or data
drive.
`--patch` and `--patch-out` wait for finalized patch artifacts, download them
through the task artifact endpoint, and fail nonzero on missing, empty, or
failed patches. `--no-stream` waits without progress output; `--detach` returns
the task id immediately.
`schedule add/list/remove` manages queue-backed scheduled tasks through the same
gateway and supervisor queue; schedules use standard 5-field cron, host-local
timezone by default, and a single catch-up run after downtime.
Benchmark helpers live under `devtools/benchmarks/`. They are tracked
operator tooling, reviewed when touched, and kept out of runtime imports. They
prepare official benchmark inputs/runs for ProgramBench, Terminal-Bench/Harbor,
SWE-bench, SWE-bench Pro, and OSWorld logs inspection without replacing
official scoring harnesses.

You can also override the bind address and port:

```bash
ouroboros server --host 127.0.0.1 --port 9000
ouroboros --url http://127.0.0.1:9000 status
```

Available launch arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--host` | `127.0.0.1` | Host/interface to bind the web server to |
| `--port` | `8765` | Port to bind the web server to |

The same values can also be provided via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OUROBOROS_SERVER_HOST` | `127.0.0.1` | Default bind host |
| `OUROBOROS_SERVER_PORT` | `8765` | Default bind port |
| `OUROBOROS_TRUST_NONLOCAL_BIND_WITHOUT_PASSWORD` | unset | Set to `1` only for trusted Docker/Kubernetes deployments where ingress auth, VPN, a private network, or an auth proxy already protects access |

For non-localhost binds, set `OUROBOROS_NETWORK_PASSWORD` (or use the
`OUROBOROS_TRUST_NONLOCAL_BIND_WITHOUT_PASSWORD=1` escape hatch only when
ingress/VPN/private-network auth already protects the surface). The full
network bind matrix and Docker/Kubernetes deployment policy live in
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) — read that before exposing
anything beyond loopback.

The Files tab uses your home directory by default only for localhost usage. For Docker or other
network-exposed runs, set `OUROBOROS_FILE_BROWSER_DEFAULT` to an explicit directory. Symlink entries are shown and can be read, edited, copied, moved, uploaded into, and deleted intentionally; root-delete protection still applies to the configured root itself.

### Provider Routing

Settings now exposes tabbed provider cards for:

- **OpenRouter** — default multi-model router
- **OpenAI** — official OpenAI API (use model values like `openai::gpt-5.5`)
- **OpenAI Compatible** — any custom OpenAI-style endpoint (use `openai-compatible::...`)
- **Cloud.ru Foundation Models** — Cloud.ru OpenAI-compatible runtime (use `cloudru::...`)
- **GigaChat** — Sber GigaChat via the `gigachat` library, OAuth key or user/password (use `gigachat::GigaChat-3-Ultra`, etc.)
- **Anthropic** — direct runtime routing (`anthropic::claude-opus-4.8`, etc.) plus Claude Agent SDK tools

If OpenRouter is not configured and only official OpenAI is present, untouched default model values are auto-remapped to `openai::gpt-5.5` / `openai::gpt-5.5-mini` so the first-run path does not strand the app on OpenRouter-only defaults.

The Settings page also includes:

- optional `/api/model-catalog` lookup for configured providers
- centralized Secrets storage for API keys, bridge tokens, passwords, and future skill-requested keys
- a refactored desktop-first tabbed UI with searchable model pickers, segmented effort controls, task-result review mode, masked-secret toggles, explicit `Clear` actions, and local-model controls

### Run Tests

```bash
make test
```

---

## Build

### Docker (web UI)

Docker is for the web UI/runtime flow, not the desktop bundle. The container binds to
`0.0.0.0:8765` by default, and the image now also defaults `OUROBOROS_FILE_BROWSER_DEFAULT`
to `${APP_HOME}` so the Files tab always has an explicit network-safe root inside the container.

> **Browser tools on Linux/Docker:** The `Dockerfile` runs `playwright install-deps chromium webkit`
> (authoritative Playwright dependency resolver) and `playwright install chromium webkit` so
> `browse_page` and `browser_action` work out of the box in the container. For source
> installs on Linux without Docker, run:
> `python3 -m playwright install-deps chromium webkit` (requires sudo / distro package access).

Build the image:

```bash
docker build -t ouroboros-web .
```

Run on the default port:

```bash
docker run --rm -p 8765:8765 \
  -e OUROBOROS_NETWORK_PASSWORD='choose-a-password' \
  -e OUROBOROS_FILE_BROWSER_DEFAULT=/workspace \
  -v "$PWD:/workspace" \
  ouroboros-web
```

Use a custom port via environment variables:

```bash
docker run --rm -p 9000:9000 \
  -e OUROBOROS_SERVER_PORT=9000 \
  -e OUROBOROS_FILE_BROWSER_DEFAULT=/workspace \
  -v "$PWD:/workspace" \
  ouroboros-web
```

Run with launch arguments instead:

```bash
docker run --rm -p 9000:9000 \
  -e OUROBOROS_FILE_BROWSER_DEFAULT=/workspace \
  -v "$PWD:/workspace" \
  ouroboros-web --port 9000
```

Required/important environment variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `OUROBOROS_NETWORK_PASSWORD` | Optional | Enables the non-loopback password gate when set |
| `OUROBOROS_FILE_BROWSER_DEFAULT` | Defaults to `${APP_HOME}` in the image | Explicit root directory exposed in the Files tab |
| `OUROBOROS_SERVER_PORT` | Optional | Override container listen port |
| `OUROBOROS_SERVER_HOST` | Optional | Defaults to `0.0.0.0` in Docker |
| `OUROBOROS_TRUST_NONLOCAL_BIND_WITHOUT_PASSWORD` | Optional | See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for the trusted-network bind policy |

Example: mount a host workspace and expose only that directory in Files:

```bash
docker run --rm -p 8765:8765 \
  -e OUROBOROS_FILE_BROWSER_DEFAULT=/workspace \
  -v "$PWD:/workspace" \
  ouroboros-web
```

### Release tag prerequisite

All three platform build scripts (`build.sh`, `build_linux.sh`,
`build_windows.ps1`) refuse to package a release unless `HEAD` is already
tagged with `v$(cat VERSION)` (BIBLE.md Principle 9: "Every release is
accompanied by an annotated git tag"). The scripts call `scripts/build_repo_bundle.py`
which embeds the resolved tag into `repo_bundle_manifest.json`, so the
launcher can later verify the packaged bundle matches a real release.

Tag the current commit before running any build script:

```bash
git tag -a "v$(tr -d '[:space:]' < VERSION)" -m "Release v$(tr -d '[:space:]' < VERSION)"
```

If the tag is missing, the build script fails with a clear error instead
of producing a bundle tagged with a synthetic/placeholder value.
Builds disable Python bytecode writes at build time, then PRECOMPILE the packaged
payload (`compileall --invalidation-mode unchecked-hash`) and SEAL the resulting
`.pyc` inside the macOS signature instead of deleting them — so there is nothing
for a normal launch to write into the signed bundle, which would otherwise break
the codesign seal. Runtime entrypoints also set `PYTHONDONTWRITEBYTECODE` with an
external cache prefix as defense-in-depth.

### macOS (.dmg)

```bash
bash scripts/download_python_standalone.sh
OUROBOROS_SIGN=0 bash build.sh
```

Output: `dist/Ouroboros-<VERSION>.dmg`, containing `Ouroboros.app` and
`Install CLI.command`. The app bundle also contains
`Contents/Resources/bin/ouroboros` and `install-ouroboros-cli`.
Chromium browser tooling is bundled in the app. WebKit/iPhone browser checks
remain available through the managed Playwright cache and may download WebKit
on first `engine=webkit` use.

`build.sh` packages the macOS app and DMG. By default it signs with the
configured local Developer ID identity; set `OUROBOROS_SIGN=0` for an unsigned
local release. Unsigned builds require right-click → **Open** on first launch.

#### Optional signing & notarization (env vars)

`build.sh` honours these env overrides so the same script ships local,
shared-machine, and CI builds without forking the script:

| Env var | Effect |
|---------|--------|
| `OUROBOROS_SIGN=0` | Skip codesigning entirely (unsigned `.app` + `.dmg`). |
| `SIGN_IDENTITY="Developer ID Application: <Name> (<TeamID>)"` | Override the codesign identity. Useful for forks whose Developer ID is not the upstream default. |
| `APPLE_ID`, `APPLE_TEAM_ID`, `APPLE_APP_SPECIFIC_PASSWORD` | When all three are set, after codesign the DMG is submitted to Apple via `xcrun notarytool submit ... --wait` and stapled with `xcrun stapler staple` so receivers do not need right-click → **Open**. Missing any one falls back to "signed but not notarized" (no Apple-side ticket exists). |

**Forks: enabling signed CI builds.** The CI release flow
(`.github/workflows/ci.yml::build`) wires the build-script env vars above
from GitHub repository secrets, plus a small set of CI-only secrets that
import the Developer ID certificate into a temporary keychain on the
macOS runner. To exercise the signed-build path in a fork, configure
**all four** of the following as repository secrets (Settings → Secrets
and variables → Actions): `BUILD_CERTIFICATE_BASE64` (base64-encoded
`.p12`), `P12_PASSWORD`, `KEYCHAIN_PASSWORD` (an arbitrary passphrase
the workflow uses for its temporary keychain), and `APPLE_TEAM_ID`. Add
`APPLE_ID` + `APPLE_APP_SPECIFIC_PASSWORD` to additionally enable
notarization. If your Developer ID identity differs from the upstream
default, also set `SIGN_IDENTITY` (e.g.
`Developer ID Application: <Your Name> (<YOUR_TEAM_ID>)`). With no
Apple secrets configured the build job falls through to
`OUROBOROS_SIGN=0 bash build.sh` and ships an unsigned DMG identical to
v5.0.0 behaviour. See `docs/ARCHITECTURE.md` §8.1 and
`docs/DEVELOPMENT.md::"GitHub Actions: secrets in step-level if conditions"`
for the rationale (job-level `env:` mapping so step-level `if:` can read
`env.*`; GHA rejects `secrets.*` in step `if:`).

### Linux (.tar.gz)

```bash
bash scripts/download_python_standalone.sh
bash build_linux.sh
```

Output: `dist/Ouroboros-<VERSION>-linux-<arch>.tar.gz`, containing
`Ouroboros/bin/ouroboros` and `Ouroboros/bin/install-ouroboros-cli`.

> **Linux native libs:** The Chromium and WebKit browser binaries are bundled, but some hosts need
> native system libraries. If browser tools fail, install deps via the bundled Python
> (the bare `playwright` CLI is not on PATH in packaged builds):
> ```bash
> ./Ouroboros/python-standalone/bin/python3 -m playwright install-deps chromium webkit
> ```

### Windows (.zip)

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_python_standalone.ps1
powershell -ExecutionPolicy Bypass -File build_windows.ps1
```

Output: `dist\Ouroboros-<VERSION>-windows-x64.zip`, containing
`Ouroboros\bin\ouroboros.cmd` and `Ouroboros\bin\install-ouroboros-cli.cmd`.

---

## Architecture

Two-process desktop app. The launcher (`launcher.py`) is an immutable
PyWebView shell; it spawns `server.py`, which runs Starlette + uvicorn
plus a supervisor thread that manages worker processes. The agent core
lives in `ouroboros/`, the SPA in `web/`, the queue/process plane in
`supervisor/`, and the system prompts in `prompts/`.

For the full file-by-file structural map, the operational layer
(every API endpoint, log file, env var, state path), and the rationale
layer (the *why* for every non-trivial design decision), see
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — that is the canonical
SSOT (Bible P6) and this README only summarizes it.

### Data Layout (`~/Ouroboros/`)

Created on first launch:

| Directory | Contents |
|-----------|----------|
| `repo/` | Self-modifying local Git repository |
| `data/state/` | Runtime state, budget tracking |
| `data/memory/` | Identity, working memory, system profile, knowledge base (including `improvement-backlog.md`), memory registry |
| `data/logs/` | Chat history, events, tool calls |
| `data/uploads/` | Chat file attachments (uploaded via paperclip button) |

---

## Configuration

### API Keys

| Key | Required | Where to get it |
|-----|----------|-----------------|
| OpenRouter API Key | No | [openrouter.ai/keys](https://openrouter.ai/keys) — default multi-model router |
| OpenAI API Key | No | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) — official OpenAI runtime and web search |
| OpenAI Compatible API Key / Base URL | No | Any OpenAI-style endpoint (proxy, self-hosted gateway, third-party compatible API) |
| Cloud.ru Foundation Models API Key | No | Cloud.ru Foundation Models provider |
| GigaChat Authorization Key (or User/Password) | No | [developers.sber.ru/studio](https://developers.sber.ru/studio) — Sber GigaChat (`GIGACHAT_CREDENTIALS` + optional `GIGACHAT_SCOPE`, or `GIGACHAT_USER`/`GIGACHAT_PASSWORD`) |
| Anthropic API Key | No | [console.anthropic.com](https://console.anthropic.com/settings/keys) — direct Anthropic runtime + Claude Agent SDK |
| Telegram Bot Token | No | [@BotFather](https://t.me/BotFather) — used by the optional Telegram bridge skill |
| GitHub Token | No | [github.com/settings/tokens](https://github.com/settings/tokens) — enables remote sync |

All keys are configured through the **Settings** page in the UI or during the first-run wizard.

### Default Models

| Slot | Default | Purpose |
|------|---------|---------|
| Main | `google/gemini-3.5-flash` | Primary reasoning |
| Code | `google/gemini-3.5-flash` | Code editing |
| Light | `google/gemini-3.5-flash` | Safety checks and fast helper tasks |
| Consciousness | empty → Main | High-horizon background consciousness |
| Fallback | `anthropic/claude-sonnet-4.6` | When primary model fails |
| Claude Agent SDK | `opus[1m]` | Anthropic model for Claude Agent SDK advisory/review internals; the `[1m]` suffix is a Claude Code selector that requests the 1M-context extended mode |
| Scope Review | `openai/gpt-5.5` | Scope reviewer slot default; `OUROBOROS_SCOPE_REVIEW_MODELS` may configure multiple independent slots |
| Web Search | `gpt-5.2` | OpenAI Responses API for web search |

Task/chat reasoning defaults to `medium`. Scope review reasoning defaults to `high`.

Models are configurable in the Settings page. Runtime model slots can target OpenRouter, official OpenAI, OpenAI-compatible endpoints, Cloud.ru, GigaChat, or direct Anthropic. When only official OpenAI is configured and the shipped default model values are still untouched, Ouroboros auto-remaps them to official OpenAI defaults. In **OpenAI-only**, **Anthropic-only**, **Cloud.ru-only**, or **GigaChat-only** direct-provider mode, review-model lists are normalized automatically: the fallback shape is `[main_model, light_model, light_model]` (3 commit-triad slots) so both the commit triad and `plan_task` work out of the box. Explicit duplicate model IDs are valid reviewer slots for stochastic sampling; lower uniqueness means lower reviewer diversity, but the quorum gate counts configured slots rather than unique model IDs. Both the commit triad and `plan_task` route through the same `ouroboros/config.py::get_review_models` SSOT. OpenAI-compatible-only setups remain explicit model-selection flows because there is no single universal default model ID for arbitrary compatible endpoints.

### File Browser Start Directory

The web UI file browser is rooted at one configurable directory. Users can browse only inside that directory tree.

| Variable | Example | Behavior |
|----------|---------|----------|
| `OUROBOROS_FILE_BROWSER_DEFAULT` | `/home/app` | Sets the root directory of the `Files` tab |

Examples:

```bash
OUROBOROS_FILE_BROWSER_DEFAULT=/home/app ouroboros server
OUROBOROS_FILE_BROWSER_DEFAULT=/mnt/shared ouroboros server --port 9000
```

If the variable is not set, Ouroboros uses the current user's home directory. If the configured path does not exist or is not a directory, Ouroboros also falls back to the home directory.

The `Files` tab supports:

- downloading any file inside the configured browser root
- uploading a file into the currently opened directory

Uploads do not overwrite existing files. If a file with the same name already exists, the UI will show an error.

---

## Commands

Available in the chat interface:

| Command | Description |
|---------|-------------|
| `/panic` | Emergency stop. Kills ALL processes, closes the application. |
| `/restart` | Soft restart. Saves state, kills workers, re-launches. |
| `/status` | Shows active workers, task queue, and budget breakdown. |
| `/evolve` | Toggle autonomous evolution mode (on/off). |
| `/review` | Queue a deep self-review: sends a generated repository atlas plus full core memory artifacts (identity, scratchpad, registry, WORLD, knowledge index, patterns, improvement-backlog) to a 1M-context model for Constitution-grounded analysis. The atlas raw-inlines selected protected/central files (ranked by import-graph centrality), accounts for every tracked path in its manifest, and excludes vendored libraries and operational logs; the in-prompt omitted-files summary is bounded, with full per-file coverage persisted in the atlas manifest. The assembled prompt is sized to an input limit that reserves output headroom inside the 1M window (window minus output reserve and tokenizer margin); if assembly overshoots, the pack retries with a compact atlas manifest and then a deterministic tighter rebuild, and only fails with an explicit error if even the shrunk pack cannot fit. |
| `/bg` | Toggle background consciousness loop (start/stop/status). |

The same runtime actions are also exposed as compact buttons in the Chat header. All other messages are sent directly to the LLM.

---

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
for the contributor workflow. The project rules remain in `BIBLE.md`,
`docs/ARCHITECTURE.md`, `docs/DEVELOPMENT.md`, and `docs/CHECKLISTS.md`;
the contribution guide only routes to those sources.

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| 6.36.2 | 2026-06-17 | **feat(process-custody): reap skill-companion orphans (log-only).** `reap_orphaned_processes` gains a companion-aware branch: a skill companion (daemon scope, `purpose companion:<skill>:<name>`) is reaped when its owning skill is uninstalled OR the ledger entry is from a foreign (dead) server generation — closing the gap where a stale companion left by a previous generation lingered and could block re-spawn on a port conflict (in-process `stop_skill`/`panic_kill_all` only cover the live generation). Ships **log-only** (`enforce_companion_reap=False` emits a `process_would_reap` event instead of killing); **fail-safe** — an unknown or momentarily-empty installed-skill set coalesces to keep-all (never a mass-kill), strict `(pid, start_time, cmd_sha256)` fingerprint, and same-session companions of installed skills are always kept so the live `CompanionSupervisor` stays their sole owner. `server.py` derives the installed-skill set from disk for the reaper. |
| 6.36.1 | 2026-06-17 | **fix(build): Windows release build — tolerate `compileall` per-file failures.** The v6.36.0 macOS-signing bytecode-seal step (`compileall`) returns a non-zero exit when any single bundled file fails to compile (the `python-standalone` ships a known tab/space-broken Tcl/Tix `WmDefault.py` that Ouroboros never imports). On macOS/Linux that was already tolerated via `|| true`; the Windows `.ps1` lacked the equivalent, so under pwsh 7.4+ (`$PSNativeCommandUseErrorActionPreference`) the native non-zero exit aborted the build. The Windows build now neutralizes Stop-on-native-error around `compileall` and resets `$LASTEXITCODE` (parity with the POSIX `|| true`) — the rest of the tree is still sealed for start-speed/signature parity. macOS/Linux builds were unaffected and already green. |
| 6.36.0 | 2026-06-17 | **feat(core): boundary resilience, unified terminalization, reviewer-slot SSOT, acceptance feedback, macOS signing — real-usage meta-fixes from a terminal-bench forensic audit.** Provider boundary: an OpenRouter HTTP-200 whose BODY carries a transient provider error (429/5xx) is no longer misread as a `finish_reason=null` "incomplete response" — the transport detects the typed body-error and reroutes ONCE to a HEALTHY endpoint of the SAME model (strips replayed `reasoning_details` + drops the `allow_fallbacks=false` provider pin), never cross-model. Unified terminalization: provider-death joins the same honest best-effort shelf as deadline/budget/round-limit (one tool-less final that benefits from the reroute → `best_effort`; else the last assistant text is salvaged) instead of discarding the workspace with a bare error string. Reviewer-slot SSOT: an ARBITRARY configured reviewer count is honored everywhere (`config.adaptive_quorum`, no `<2` hard gates / `[:3]` cap) — a single configured reviewer runs as a loud `single_reviewer_no_diversity` degraded mode (plan_task, skill trust-gate, commit/scope), while configured-≥quorum-but-fewer-responded stays a loud infra quorum failure. Acceptance review now feeds the agent a compact anti-derailment improvement capsule (tier + ≤3 actions + coach) in BOTH `auto` and `required` — one bounded pass, full verdict on the objective axis. Tool robustness: binary stdout decodes tolerantly (`errors='replace'`) at every command boundary, a present-but-unchanged declared output is cosmetic not blocking, and `vlm_query` reads the active task workspace. macOS: the signed/notarized `.app` precompiles + SEALS its bytecode (eliminating the runtime `__pycache__` writes that broke the codesign seal → AppTranslocation) with a global `PYTHONDONTWRITEBYTECODE` + whitelist coverage for curated-env embedded-python spawns. |
| 6.35.1 | 2026-06-17 | **feat(core): real-usage meta-fixes from a benchmark forensic audit.** Honest task-acceptance review: `outcome_tier`/`completion_coach` are now REQUIRED reviewer JSON keys (reviving the best_effort completion-coach lexicon), the reviewer judges EVIDENCE INDEPENDENCE (whether passing evidence comes from tests the agent itself authored) and separates an ENVIRONMENT fault from a wrong DELIVERABLE, and host-forced `required` review is **label-only** — it records the verdict/tier on the objective axis without the meta-essay re-loop that previously tanked metrics (the agent still gets feedback when it self-calls review in the default `auto`); a single parse-degraded reviewer slot no longer poisons a clean quorum PASS; the turn diff (tracked + untracked new files) is fed to the reviewer as a host-owned structural fact. Tool ergonomics for any external workspace: `search_code`/`list_files`/`query_code`/`read_file`/`write_file` accept absolute and redundant-root-prefix paths inside the active root (`/app/x` and `app/x`), and `run_command`/`run_script` accept a per-call `timeout_sec` (alias `timeout`) that lifts the outer tool-execution cap so long builds aren't cut off at the static 360s entry. Outcome honesty: an ignored one-shot `run_command`/`run_script` non-zero exit becomes a non-degrading cosmetic record, with a structural `residual_tool_errors_without_review` warning surfaced when the objective was never judged. Real-usage `workspace.patch` hygiene: untracked build binaries, oversize blobs, and junk artifacts are excluded (recorded in the manifest, never silently lost). |
| 6.34.0 | 2026-06-16 | **feat(core): multi-task chat steering + resilience + v6.33.0 review carryover.** A busy-chat decision turn can now STEER the right running task instead of spawning a duplicate: it sees the chat's running tasks as structural runtime context (`current_chat.running_tasks`) and the new LLM-first `steer_task(task_id, message)` delivers to that task's owner-mailbox (idempotent, fail-visible on a stale target) — generalizing to N concurrent tasks (the agent's judgment picks the target, no keyword gate; a project-room message also defaults to that project). Resilience: extension skills decide the ctx calling-convention on the RAW handler (no spurious-ctx TypeError for keyword-only handlers); the supervisor intakes new messages EARLY and a dedicated liveness watchdog surfaces a supervisor-loop stall OR a heartbeat-silent in-process chat turn (owner alert + `/restart` hint) instead of silent hours; WebSocket broadcasts go out concurrently (one slow client can't head-of-line) and chat-history jsonl parses off the event loop. Carryover fixes for the v6.33.0 review's blocked-8: the P3 scope-review floor is owner-only + audited (`POST /api/owner/scope-review-floor`, merge-skipped from generic settings, guarded on the shell/browser/SAFETY channels); Max context mode is enforced at point-of-USE and point-of-BUILD (fail-closed to Low when the active route — remote OR local n_ctx — no longer confirms ≥1M, read-only; `USE_LOCAL_MAIN` routes the gate to local n_ctx; `switch_model` refuses a sub-1M route while the transcript is max-sized); a short ephemeral decision turn runs a default-deny read/decision ALLOWLIST (no durable/control/review/skill/shell or extension·MCP tools, and it leaves no durable task record); the external-shell secret guard catches relative interpreter-string paths; the named-project how-to moves from SYSTEM.md into the tool description. Ratified extras: `op=structural` code intelligence is polyglot via tree-sitter (Go/Rust/Java/… + a visible `structural_unavailable` marker); an OpenAI-compatible `/models` capability probe; Settings Max-save shares the capability-ack flow. New surface: `POST /api/owner/scope-review-floor`, `steer_task` tool, `OUROBOROS_SUPERVISOR_LIVENESS_DEADLINE_SEC`. |
| 6.33.0 | 2026-06-15 | **feat(core+ui): Capability-Evidence context modes, project task-bindings, and the WS11 UI/UX round.** Context window is no longer a static per-model table: every window claim is sourced, route-fingerprinted **Capability Evidence** (provider `/models` metadata, local n_ctx, or an owner acknowledgement) with a status (`confirmed`/`asserted`/`unprobeable`/`failed`), and Max context mode is fail-closed — it requires ≥1M confirmed/asserted evidence for the active route. Changing the model while Max is on stays friction-free: the change succeeds and context auto-downgrades to Low with a plain notice when the new route can't be confirmed ≥1M, but a genuine **no-connection during the probe is an error** (the model is not saved), and a transient provider outage never erases a prior confirmed record. Multi-project: a main-chat task converts to a project in one click, **auto-named from its own objective** (no prompt, no extra LLM call; in-flight conversions read the live queue); project-chat follow-up tasks **bind to their project** so the main chat shows no stray "turn into project" button and instead a calm pointer that opens the project panel; a converted card becomes a calm indigo **project identity** (no red "error" look); per-project **unread** dots sort active projects to the top (server-stored last-viewed); the project **status / sleep-wake lifecycle was removed** (`/api/projects` is list/create only). UI polish: oval (pill) composer with vertically-centered controls, and **per-thread chat scroll is restored** on tab/panel switch instead of jumping to the top. Also: real `deadline_at` finalization + advisory pacing, tree-sitter code intelligence for non-Python symbols (`query_code op=digest`), reflection faculty-atrophy doctrine, and assorted WS9 tool fixes. New surface: `POST /api/owner/capability-ack`, `ouroboros/capability_evidence.py`, `data/state/capability_evidence.json`. |
| 6.32.2 | 2026-06-14 | **fix(ci/ui): finish the v6.32.0 UI-smoke alignment so the release build goes green.** Relax the mobile composer input-width assertion — the redesigned 390px row places the attach button, text input, and Send inline (chips ride above), so the input is naturally below the old desktop-era 300px target; the smoke now asserts a usable width and that the input never overlaps Send. Verified live + by adversarial multi-model review. Test-only; no runtime/contract change. |
| 6.32.1 | 2026-06-14 | **fix(ci/ui): complete the v6.32.0 multi-project release — align the Playwright UI smoke suite with the shipped composer/nav redesign.** The desktop composer smoke now asserts the Consilium + Low/Max chips render ABOVE the text input (Send stays inside the input row), and the mobile smoke opens the new sidebar drawer (`[data-mobile-nav-toggle]` → `#primary-sidebar.open`) and navigates via `data-nav-page` rows instead of the removed `data-page` rail. Test-only fix (verified live + by an adversarial multi-model review); no runtime/contract change, the v6.32.0 feature set is unchanged. |
| 6.32.0 | 2026-06-13 | **feat(multi-project): штаб и проекты — one agent, parallel durable projects.** The single agent gains owner-facing projects while identity/constitution/evolution stay unified: a durable registry (`data/state/projects.json`, boot-reconciled, never age-pruned) with deterministic per-project chat ids; the conversation lane stays free while real work routes to first-class pooled tasks via the LLM-first `promote_chat_to_task` tool; project chats ride the same WebSocket with `chat_id`-stamped frames, per-thread history (`/api/chat/history?chat_id=`), and owner-mailbox steering of the project's running task. Per-project memory grows journal (`journal_write`/`journal_read`: start/checkpoint/blocked/done), workpad (`workpad_*`), and bounded context injection beside project knowledge; finished project tasks append journal rows and emit sanitized `project_digest` observations to consciousness (no raw-fact leaks). One-writer-per-project lease in task assignment (subagent swarms exempt); agent-requested restarts drain heartbeat-fresh tasks up to `OUROBOROS_RESTART_DRAIN_MAX_SEC`; optional invisible-git working folders via the genesis machinery; `/api/projects` list/create/sleep/wake. UI: sidebar Projects list with a visually distinct Main chat; a project opens as a right split panel (desktop) / overlay (mobile) hosting a full chat instance (`createChatInstance` factory — per-instance DOM/storage/thread filter; global evolve/panic/budget controls stay main-chat-only). |
| 6.31.0 | 2026-06-13 | **feat(skills): unix_computer_use overhaul + native launcher-seed trust.** The bundled desktop skill is renamed `computer_use` → `unix_computer_use` (Windows stays a future separate skill) and gains coordinate normalization (screenshots downscale to WXGA and persist the exact image→input transform; input tools auto-remap, `raw=true` bypasses), Wayland backends (`grim`/`ydotool`/`wtype` beside X11 `xdotool`/`scrot`), new actions (`left_click_drag`, `mouse_down`/`mouse_up`, `triple_click`, `hold_key`, `cursor_position`, `wait`), a real set-of-marks `ax_tree` on macOS (numbered interactive elements of the frontmost window with honest degradation), and extended cross-platform key aliases. Launcher-seeded native skills now carry a named, hash-pinned, audited trust verdict (`OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS`, default-on; CHECKLISTS §Skills): the launcher stamps `review.json` `status=clean` (`repo_commit_gate`/`native_seed`) at bootstrap/new-seed/version-resync because the payload bytes passed the repo triad+scope gate; zero-grant skills (tool/subprocess surface only) also auto-enable, externally-facing ones (e.g. weather: net/route/widget) stay disabled. Lifecycle control files (`.seed-origin`, top-level, native bucket only) leave the payload hash with a one-shot re-pin migration for unedited legacy-hash reviews. Upgrade note: installs that previously received the old `computer_use` seed keep it as a reclassified non-launcher skill next to `unix_computer_use` — delete `data/skills/native/computer_use/` to drop the duplicate toolset. |
Older releases are preserved in Git tags and GitHub releases. Older 6.x rows, the 5.2.0 through 5.33.0-rc.6 rows, and former `4.0.0` rows are rolled off to respect the P9 changelog cap; their full bodies remain at their git tags.

---

## License

[MIT License](LICENSE)

Created by [Anton Razzhigaev](https://t.me/abstractDL) & Andrew Kaznacheev
