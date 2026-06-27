# Ouroboros

[![GitHub stars](https://img.shields.io/github/stars/razzant/ouroboros?style=flat&logo=github)](https://github.com/razzant/ouroboros/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![macOS 12+](https://img.shields.io/badge/macOS-12%2B-black.svg)](https://github.com/razzant/ouroboros/releases)
[![Linux](https://img.shields.io/badge/Linux-x86__64-orange.svg)](https://github.com/razzant/ouroboros/releases)
[![Windows](https://img.shields.io/badge/Windows-x64-blue.svg)](https://github.com/razzant/ouroboros/releases)
[![OuroborosHub](https://img.shields.io/badge/OuroborosHub-skills%20marketplace-8A2BE2.svg)](https://github.com/razzant/OuroborosHub)
[![Version 6.50.2](https://img.shields.io/badge/version-6.50.2-green.svg)](VERSION)

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
SWE-bench, SWE-bench Pro, GAIA, and OSWorld logs inspection without replacing
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
| Heavy | empty → Main | Strong acting/coding lane (`OUROBOROS_MODEL_HEAVY`; renamed from `Code`, empty falls back to Main) |
| Light | empty → Main | Safety checks and fast helper tasks (`OUROBOROS_MODEL_LIGHT`, empty falls back to Main) |
| Vision | empty → Main | Caption/VLM lane (`OUROBOROS_MODEL_VISION`, empty falls back to Main for remote routes; local/blind routes need an explicit reachable vision slot for caption fallback); image input routing is controlled by `OUROBOROS_IMAGE_INPUT_MODE=auto|caption|inline|off` |
| Consciousness | empty → Main | High-horizon background consciousness |
| Fallbacks | `anthropic/claude-sonnet-4.6` | Comma-separated cross-model fallback chain when the primary fails (`OUROBOROS_MODEL_FALLBACKS`) |
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
| 6.50.2 | 2026-06-27 | **fix(scope-review): preserve blocking scope-review authority while handling false-window overflow.** The scope reviewer window probe now uses the actual scope-review slot route instead of the active main route, recognizes direct-provider and explicit OpenRouter spellings of the shipped `openai/gpt-5.5` reviewer, and only downgrades opaque gateway 400s to `budget_exceeded` when independent size evidence points to real oversize. Off-default reviewers without Capability Evidence now surface advisory-only under the 1M floor until owner-acked, instead of inheriting false 1M trust. Regression tests cover direct/default reviewer identity, local-main plus remote-scope routing, and large non-oversize provider errors staying fail-closed. |
| 6.50.1 | 2026-06-27 | **docs(prompt): sync resident SYSTEM.md with the v6.50.0 reconciliation layer.** The system prompt now points agents at the live `schedule_subagent` schema for capability declarations, names structured `delegation_constraint` back-pressure and parent-only `override_delegation_constraint`, and calls out the runtime filesystem affordance map without duplicating schema-only directive details. No release tag, build, or stable promotion was produced for this patch-only branch update. |
| 6.50.0 | 2026-06-27 | **feat: reconciliation layer for reliable subagent swarms plus SWE-bench Pro adapter hardening.** Runtime: the agent now reasons forward from a live filesystem affordance map, reconciles declared child capabilities and delegation constraints before scheduling, queues excess children instead of dropping them at the active-subagent cap, prevents clean finalization while delegated children are still running, and fixes a light-mode shell false positive by classifying interpreter writes by resolved write targets rather than repo-path mentions. Bench/devtools: the SWE-bench Pro adapter hardens Alpine/musl install-in-image transport (no Python ABI drift, pyexpat/pip/import fail-fast, unsupported Playwright dropped, no read-only editable install), captures patches without pre-existing untracked task-image fixtures, and distinguishes permanent infra non-runs from retryable transient failures. Historical note: v6.48.0/v6.49.0 were pushed without annotated tags; v6.50.0 resumes the release tag stream by operator decision. |
| 6.49.0 | 2026-06-27 | **feat: generalize same-model provider failover to all verified-portable reasoning families (OpenRouter rate-limit resilience).** Extends v6.48.0 beyond Anthropic: the `allow_fallbacks=false` reasoning-continuity pin is now lifted for every family whose replayed reasoning signature is cross-provider portable on OpenRouter — Anthropic (Anthropic/Bedrock/Vertex/Azure), Gemini (Vertex/AI-Studio), and OpenAI encrypted reasoning (OpenAI/Azure), each verified live by a same-model replay probe forcing each sibling provider. The prod MAIN (gpt-5.5) and FALLBACK (gemini) now fail over to a healthy sibling endpoint under an upstream rate-limit instead of surfacing it, while OpenRouter's sticky routing keeps the prompt cache warm on the primary (no cache thrash, no explicit provider ordering needed). Unverified/exotic families (e.g. `z-ai/glm`, `deepseek`) keep the conservative pin; the reactive signature-rejection `400` strip-and-retry remains the safety net for every family. New SSOT predicate `_reasoning_signature_portable_across_or_providers`. |
| 6.48.0 | 2026-06-27 | **feat: Anthropic same-model provider failover keeps reasoning continuity (OpenRouter resilience).** A replayed-reasoning OpenRouter request no longer pins `provider.allow_fallbacks=false` for Anthropic-family models: Anthropic thinking-block signatures are cross-platform compatible (Anthropic API / Bedrock / Vertex / Azure — confirmed against Anthropic's extended-thinking docs and a live same-model OpenRouter replay probe), so a same-model failover preserves reasoning continuity and no longer surfaces one upstream's rate-limit when a healthy sibling endpoint could serve the turn. Non-Anthropic families (e.g. OpenAI / Gemini encrypted reasoning) keep the conservative pin, and the reactive signature-rejection `400` strip-and-retry stays the safety net for every family. |
| 6.47.1 | 2026-06-26 | **fix: verify-before-done observability + ledger grounding-status accounting (v6.47.0 fix-forward).** The verification ledger's `has_failures` summary no longer mis-reads a SUCCESSFUL grounding receipt as a failure: the `observed` (artifact_observation) and `declared` (honest no_visible_machine_contract) statuses join the success allow-list via the `_RECEIPT_GROUNDING_STATUSES` SSOT, in both `build_verification_ledger` and `refresh_verification_ledger_artifacts`. The `receipt_absent` / `expected_output_ungrounded` objective-axis warning now reaches the `task_eval` / `task_metrics` event stream (where day-one monitoring reads it), not only the stored task_result: the flag is applied once in `emit_task_results` before the events emit, and the single flagged `loop_outcome` is threaded into `_store_task_result` (single source, no second derive). Devtools: a GAIA `--sample-id` re-run selector for replaying a chosen subset of tasks, and root-anchored `.gitignore` entries so the in-repo solver's downloaded attachments / OCR scratch stop polluting the working tree. |
| 6.47.0 | 2026-06-26 | **feat: verify-before-done flagship, deep cooperative subagents, workspace-aware code-intel, and the skill-publish desync fix.** Reliability: a new core `verify_and_record` tool runs the agent's declared verification check on the host and writes a durable host-attested receipt (contract kinds: visible_verifier / explicit_command / explicit_metric / artifact_observation / no_visible_machine_contract); a one-shot verify-before-done nudge fires when a turn produced real effects but recorded no grounding, and a BINARY `receipt_absent` (FR3) / `expected_output_ungrounded` (M2 zero-grounding, gated on the typed `expected_output`) objective-axis warning surfaces an unverified-but-clean turn WITHOUT ever downgrading it (anti-oscillation); a deadline-flush prompt salvages a time-boxed deliverable at the tightest milestone. Capability: `query_code` runs read-only structured code intelligence over an external `user_files` target (e.g. a benchmark `/app`) with search_code-shape guards and bounded symlink-safe structural walks; external-workspace HOST process commands scrub the Ouroboros repo from `PYTHONPATH` so a target's `import web`/`server` can't shadow-import the agent (a `docker_exec` backend already runs in the container's own environment); the artifact-audit nudge is now effect-based; deep COOPERATIVE acting subagents build together in ONE host-minted shared git tree (external_workspace), with a depth-aware reservation so a wide first level can't starve depth. UX: skill→hub publish eligibility is one SSOT predicate the gateway serializes (`submit_hub`) and the Skills card renders — a no-blocker (clean OR advisory-only warnings) review now enables Submit in the UI, ending the clean-only desync. Bench: the SWE-bench Pro grader prints a RAW Pass@1 headline and keeps all contamination/benchmark-defect analysis in a separate diagnostic `CONTAMINATION_AUDIT.md` that never adjusts the score (gold is never shown to the solver). New surface: `verify_and_record`, `ouroboros/skill_publish_eligibility.py`, `submit_hub` gateway field. |
| 6.46.0 | 2026-06-25 | **feat: GAIA-forensic hardening — generalizable runtime fixes surfaced by a benchmark audit, plus an empirical context-window probe.** Runtime: budget exhaustion now TERMINALIZES undispatched queued tasks (`fail_tasks`, `reason_code=budget_exhausted` + event/notify) instead of a silent never-dispatched PENDING freeze; `search_code` gains a wall-clock deadline across the directory walk AND the rg-batch loop (`OUROBOROS_SEARCH_CODE_WALL_SEC`); `web_search` gains a backend pin (`OUROBOROS_WEBSEARCH_BACKEND`, `ddgs` now a dependency with 429 backoff) so a fixed-model run can force pure retrieval; a config-SSOT `OUROBOROS_OR_PROVIDER` (resilience/repro presets) routes OpenRouter providers for rate-limit resilience without losing prompt-cache warmth, generalized behind a `supports_message_cache_control` predicate. Answer integrity: a tool-count-stamped latch recovers a produced `FINAL ANSWER` whose marker a later round DROPS (a deliberate post-review marker — a genuine correction — stays authoritative, and new tool work invalidates a now-stale latch), and acceptance review on the advisory task surface stops over-failing a grounded answer (majority-not-single-FAIL, no solved→DEGRADED on empty coach) so review can ground but never single-vote-DOWNGRADE a correct answer — the P3 commit/scope immune gate is untouched. Security: `user_files` honors `OUROBOROS_USER_FILES_ROOT` (jailed scratch home), and the authoritative observability blob is redacted by default (`OUROBOROS_OBSERVABILITY_KEEP_RAW` opt-in) with a broadened secret matcher. Capability: a generative context-window probe empirically confirms a route's ≥1M window from a FREE over-window reject (only on the explicit Max toggle/save, never from a possibly-paid 200), plus a cloud.ru route base_url fix. Bench: the GAIA adapter stages attachments robustly, injects the `FINAL ANSWER:` protocol, crash-isolates per-sample timeouts, and exposes per-config knobs. |
| 6.45.5 | 2026-06-24 | **fix: CI expectation compatibility for credential-aware tool gating.** Clear inherited timeout env in shell timeout tests, seed web-search availability where resource-gate behavior is under test, and accept central `CAPABILITY_UNAVAILABLE` for Claude Code auth-gate expectations. |
| 6.45.4 | 2026-06-24 | **fix: Terminal-Bench timeout smoke expectation after safety-buffer accounting.** Align the hostile-workspace runner test with the adapter’s effective task-timeout buffer (`900 -> 870`) while preserving the generated task timeout contract. |
Older releases are preserved in Git tags and GitHub releases. Older 6.x rows, the 5.2.0 through 5.33.0-rc.6 rows, and former `4.0.0` rows are rolled off to respect the P9 changelog cap; their full bodies remain at their git tags.

---

## License

[MIT License](LICENSE)

Created by [Anton Razzhigaev](https://t.me/abstractDL) & Andrew Kaznacheev
