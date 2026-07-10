# Ouroboros

[![GitHub stars](https://img.shields.io/github/stars/razzant/ouroboros?style=flat&logo=github)](https://github.com/razzant/ouroboros/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![macOS 12+](https://img.shields.io/badge/macOS-12%2B-black.svg)](https://github.com/razzant/ouroboros/releases)
[![Linux](https://img.shields.io/badge/Linux-x86__64-orange.svg)](https://github.com/razzant/ouroboros/releases)
[![Windows](https://img.shields.io/badge/Windows-x64-blue.svg)](https://github.com/razzant/ouroboros/releases)
[![OuroborosHub](https://img.shields.io/badge/OuroborosHub-skills%20marketplace-8A2BE2.svg)](https://github.com/razzant/OuroborosHub)
[![Version 6.61.4](https://img.shields.io/badge/version-6.61.4-green.svg)](VERSION)

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
- **Layered Safety** — Hardcoded sandbox blocks writes to safety-critical files and mutative git via shell; an explicit per-tool policy map decides which built-ins skip the LLM check; everything else goes through a single light-model safety call under the default `OUROBOROS_SAFETY_MODE=full` (the owner-only `light`/`off` coverage modes wave LLM checks through with durable audit events — the deterministic layer never turns off). The fail-open contract, protected-path guard, and full provider-mismatch matrix live in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §Safety system and [`prompts/SAFETY.md`](prompts/SAFETY.md).
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

If OpenRouter is not configured and only official OpenAI is present, untouched default model values are auto-remapped to `openai::gpt-5.5` / `openai::gpt-5.4-mini` so the first-run path does not strand the app on OpenRouter-only defaults.

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
| Scope Review | `anthropic/claude-fable-5` | Scope reviewer slot default; `OUROBOROS_SCOPE_REVIEW_MODELS` may configure multiple independent slots |
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
| 6.61.4 | 2026-07-09 | **fix: answer-protocol finalization honesty — tier-token answers rejected, marker nudge no longer requires expected_output.** Two GAIA v6.56.0 trace-verified losses were harness-shaped, not capability: (1) after an acceptance downgrade the agent shipped `FINAL ANSWER: blocked_with_evidence` — the snake_case outcome-tier ledger identifiers (`best_effort`/`blocked_with_evidence`) are now structurally rejected by `extract_final_answer` (count as missing → the salvage/nudge path recovers a real answer; `solved` stays extractable as an ordinary English word), and both the answer-protocol context rule and the improvement capsule now say tier words are ledger metadata, never the answer; (2) a last-round refusal finalized with an EMPTY typed answer despite 24 tool calls of real research because the P2 final-marker nudge ALSO required a declared `expected_output` — GAIA-shaped contracts carry the question in `objective` with `expected_output` empty, so the only salvage surface was suppressed; the protocol gate (`answer_protocol_active`) is sufficient on its own. Bench canon untouched: both tasks stay scored as-is in the v6.56.0 table. |
| 6.61.3 | 2026-07-09 | **fix: project-room chat lens — the room's chat lane looks at the project folder.** The robot-room incident: a folder-room's DIRECT-CHAT lane resolved `"."` against the system repo while the room fact named the project folder, and the agent narrated the wrong tree as the project (affordance-context incoherence). The chat lane of a folder-room now re-points `active_workspace` READS (`read_file`/`list_files`/`search_code`/`query_code`) and the DEFAULT shell cwd at the room's registered `working_dir` (`project_room_lens_dir`, keyed strictly on direct-chat + no own workspace + a host-verified room dir; pooled/workspace/subagent/headless tasks, benchmarks, and file-less rooms are byte-identical to before). Default-root writes (`write_file`/`edit_text`/default-cwd `claude_code_edit`) in a folder-room return a typed `ROOM_WRITE_VIA_TASK` refusal pointing at `promote_chat_to_task` — mutations stay with promoted tasks, so the one-writer lease is never contended by the chat lane; the self-repo stays one explicit root away (`root="system_repo"`, explicit shell cwd). The first default-cwd shell command disclosed its room cwd; a set-but-broken working_dir rides the room fact as a loud `working_dir_warning`. The room fact's stated rule and the actual tool surface come from the SAME resolver and are pinned by a coherence invariant test. Sidebar: the New Project "+" no longer overlaps the projects count (the header toggle reserves right-side space; verified geometrically with a non-empty badge). |
| 6.61.2 | 2026-07-09 | **fix: 3-OS test portability for the v6.61.1 hardening suites.** Four Windows-runner failures in the release matrix, all test-side: the artifact-observation traversal probe now escapes to the filesystem ROOT on every OS (Windows nests pytest tmp ~7 levels under the user home, so six `..` landed INSIDE home where the deliberate user_files read lane makes the probe an honest miss, not a refusal); the bytes_equal confinement test jails `OUROBOROS_USER_FILES_ROOT` into the workspace so its refusals are deterministic cross-OS; the coop-checkpoint test pins `encoding="utf-8"` on the git-log read (the ANSI code page mangled the em-dash in the commit subject); the broken-working_dir loud-fail test removes a git tree via a chmod-and-retry `rmtree` handler (read-only object files raise WinError 5). Runtime code unchanged. |
| 6.61.1 | 2026-07-09 | **fix: adversarial-review round-1 hardening.** `bytes_equal` operands are CONFINED like every other artifact-path surface (`_confine_artifact_path` + protected-artifacts `read_bytes` denial on both files; executor operands must be workspace-relative — no absolute/`..` oracle over hidden grader files) and the mode is rejected for non-run contract kinds instead of silently ignored. The Q7 effort-clamp disclosure is actually emitted: a learned-ceiling clamp records `reasoning_effort_clamped={requested, applied, reason}` into that call's usage event on all lanes (OpenRouter/direct/Anthropic), `_record_effort_ceiling` floors at "low" so a rejection of the lowest thinking tiers can never poison a route to `none`, and Anthropic-direct maps our `minimal` to the provider floor `low` (its documented set has no minimal). The accidental MagicMock test artifact `logs/events.jsonl` (committed mid-sprint by a polluted test run, so it never appears in the base..release diff) is deleted again in history and repo-root `/logs/` is gitignored (anchored). SYSTEM.md's subagent-yield line no longer carries the line-wrapped marker phrase and both pinning tests assert on whitespace-normalized text. `POST /api/projects` returns 409 `project_exists` for an existing id with a requested source (checked BEFORE any clone/attach side effect); `GET /api/fs/dirs` carries an honest `truncated` flag (>500 children) surfaced in the picker; the chat parent card's task_done meta shows own cost `+children` rollup (README/ARCH parity); the Safety skip-counter labels its recent-events window. |
| 6.61.0 | 2026-07-09 | **feat: adaptive planning — plan_class, tiered reviewer docs, task-fit scouts (Phase 5).** `plan_task` gains an agent-declared `plan_class` (self_mod | external | creative | research) with STRUCTURAL escalation: files_to_touch resolving under the Ouroboros system repo force self_mod (a path fact, never keywords — P5). self_mod keeps today's full governance pack and the explicit `context_level` contract, untouched. Non-self_mod plans (external codebases, creative deliverables, research) get an owner-approved governance evolution (quiz 19, DEVELOPMENT.md Core-Governance table + prose updated in the same commit): reviewers keep BIBLE.md and DEVELOPMENT.md in full but receive ARCHITECTURE.md as the lossless navigation map (`generate_doc_nav_map` — every section + line range, full sections on demand), `context_level` may be omitted (defaults to minimal — the generated Atlas is repo archaeology an external plan needs only on request), and the reviewer prompt carries the class so plans are judged against their own domain. Planning scouts are class-framed: self_mod scouts keep the repo-archaeology emphasis, external/creative/research scouts are steered to the plan's own domain (requirements coverage, verification strategy, sources, design/content quality) — never Ouroboros internals by default; the swarm fingerprint carries the class so a re-run under a different class never resumes the other's handoffs. |
| 6.60.0 | 2026-07-09 | **feat: answer protocol by contract, blocking widening, bytes-equal verification (Phase 4).** The `FINAL ANSWER` doctrine leaves the global prompt: new additive `task_contract.answer_protocol` ("" | "final_answer_line", `normalize_answer_protocol`, `/api/tasks answer_protocol=` + CLI `--task-metadata-json`, inherited by subagents) — when declared, the runtime context carries the protocol instruction (with the opt-in `CANDIDATES:` ambiguity block), the P2 marker nudge and the pacing salvage phrases activate; without it, ordinary chat/self tasks never see marker prompting (SYSTEM.md rule + CANDIDATES section removed; the latch/extractor/typed `final_answer` stay unconditional; `final_answer_missing_sentinel` keys on the typed payload so latch-recovered answers are not "missing"; the no-op nudge keys on expected_output semantics). GAIA declares the field; TB/SWE-Pro/PB deliberately do not (state/patch/code deliverables). The web UI renders the marker as a labelled Answer chip. Blocking widening (S1-lite): under required+blocking, HIGH-severity contributing findings with a concrete recommendation become typed obligations WHEN the aggregate verdict is failing (signal FAIL or tier blocked_with_evidence) — critical-only on PASS; the dead `verdict_is_advisory` policy key is removed; the acceptance checklist asks the scope-cut question explicitly; SWE-Pro settings_base flips to blocking (PB/TB already were). Verification: `expected_match="bytes_equal"` compares two declared files byte-for-byte after the check on the same surface as the check (executor cmp / host chunked read) with a bounded first-divergence hexdump in the receipt; `schedule_subagent` documents the independent-verifier pattern (read-only memory_mode=empty child fed only the deliverable + acceptance criteria). |
| 6.59.0 | 2026-07-09 | **feat: project entry points — attach, clone, New Project UI, agent one-liner (Phase 3).** `POST /api/projects` creates from ONE of four sources: `path=` attaches an existing owner folder (new `project_sources.py`: resolved-realpath validation — exists/directory/not-home-root/no repo-data overlap; opt-in `init_git` makes an attach-snapshot commit with a local identity, NEVER auto-init), `git_url=` clones server-side into the durable projects root (atomic tmp→rename, `GIT_TERMINAL_PROMPT=0` + ssh BatchMode, typed `auth_required` for private repos), `with_workspace` provisions a genesis folder, or none — a file-less project. `provenance` (attached|cloned|genesis|none), `clone_url`, and an automatic `trusted_at` (notification trust model: attaching IS the owner's grant) land as additive registry facts. New Project "+" dialog in the sidebar (name + 4 sources, server-side home-confined directory browser via `GET /api/fs/dirs` — works in web/Docker, honest "agent gets write+shell here" note); per-row kebab: rename (`POST /api/projects/{id}/update`), hide (presentation-only `project_hidden` ui-preference), delete (`POST /api/projects/{id}/delete` — un-registers + unbinds, the folder and memory store are never touched). The agent gets the same power in one move: `promote_chat_to_task(source=<path or git URL>)` attaches/clones through the same primitives, registers the folder on the project, and reports loudly — "help me debug this GitHub repo" becomes a one-liner. Owner-declined tradeoffs recorded in ARCHITECTURE so reviewers stop re-proposing them: no env secrets-scrub for workspace shells (quiz 12), PR-flow and workspace-AGENTS.md reading deferred (quizzes 14/15). |
| 6.58.0 | 2026-07-09 | **feat: projects foundation — one folder = one identity, room→workspace wiring, coop checkpoint (Phase 2).** `resolve_project_id` is registry-first: a workspace path equal to a registered project's normalized `working_dir` resolves to THAT project id (the `proj_<hash>` is minted only for unregistered folders), and every admission surface stores the resolved id, so one folder is one serialized writer lane on all entry paths; `projects.json` gains `_schema_version`. New `workspace_admission.py` SSOT: the single workspace-root validator shared by `/api/tasks` and the promote path (the promote path's degraded twin — raw unvalidated `workspace_root`, no fork/preflight — is gone), `resolve_room_workspace` (a task promoted in a project ROOM inherits the room's `working_dir` as its active workspace by default; `workspace="none"` opts out; a SET-but-broken working_dir LOUD-FAILS the task with a visible card+message — never a silent workspace-less self_modification-profile task), `compose_workspace_block` (shared `[HEADLESS_WORKSPACE]` guidance), and `bounded_workspace_preflight` (hard-capped so the supervisor event thread stays responsive). Decision turns keep their promote/steer toolset and instead see the room's `working_dir` as a structural `project_room` fact. Project-from-task §3.4 fixes: the frontend hint splits into a 60-char NAME channel and a FULL-text chat-mirror channel (the truncated «Сделай html сайтик…в…» incident), an already-finished task's final answer + artifact paths are mirrored into the new project thread, and the proactive namer gets a hard wall-clock bound (the 24-minute-late title). Coop pair: `integrate_subagent_patch` returns a SUCCESSFUL `coop_already_in_tree` no-op for a non-workspace parent whose child built in the host-minted shared tree (read-only verified), and new `coop_checkpoint.py` checkpoint-commits dirty genesis/coop trees at root finalization (credential-shaped files excluded and disclosed; owner-attached folders never auto-committed). |
| 6.57.0 | 2026-07-09 | **feat: swarm & outcome honesty + safety/effort settings (Phase 1).** `find_child_tasks` gains `scope="direct"|"subtree"`; per-node absorption/handoff use `direct`, so a childless grandchild no longer gets a false `children_unabsorbed` reminder about its parent/sibling. `verify_and_record` adds a `refused_out_of_scope` receipt status (a policy refusal, not a `fail`, never raising `has_failures`) and lets `artifact_observation` confirm a child's deliverable under the read-only `subagent_projects`/`deliverables` roots (existence/size only). Outcomes gain a `policy_denials` bucket: an unrecovered `*_blocked` refusal on ANY tool (incl. write/shell/integration) is telemetry, not an execution degrade and not a `tool_failure` headline — the site-presentation incident where `integration_blocked`/`LIST_FILES` reddened a shipped site; genuine errors still degrade, and `trace_summary` shows the honest bucket breakdown so self-learning is not poisoned. The idle heartbeat is suppressed while a descendant is progressing (not latched, so it still fires on genuine idle). `schedule_subagent` returns the child's effective profile summary (shell/writable roots/lane) to the parent and injects it into the child's start context; a capability mismatch names the correct spawn; a new recursive `cost_usd_with_children` field rolls up direct-child cost (partial-marked) on the parent card + Logs. Protected-artifact policy gains a glob carve-out (`rm -f *.out` beside a black-box ref no longer blocks; a pattern that could match the ref still does) and refusals now name the nearest allowed action. New Settings→Behavior **Safety Supervisor** card (Full/Light/Off via the audited owner endpoint, confirm-on-lower, 24h skip counter) and an `EFFORT_SCALE` SSOT (none→max) — `xhigh`/`max` added, direct-Anthropic effort mapped to adaptive-thinking `output_config.effort` (was a dead control), and a requested effort clamped down to each route's learned ceiling (disclosed, never silent). `MAX_TOTAL_FUNCTIONS` 3699→3740. |
| 6.54.4 | 2026-07-03 | **fix: review depth + verification provenance from the bench post-mortems.** New `task_pacing.py` SSOT absorbs the loop's milestone/pacing content and adds the acceptance-review budget layer: a finalization reserve (max of the grace window and `budget_profile.reserve_finalization_pct`), a budget snapshot, and two independent gates — a review may launch only above the reserve (loud `review_skipped_deadline_reserve` otherwise), and improvement passes are bounded by BOTH a pass counter and the time-above-reserve window (policies fixed/adaptive/until_deadline from the new typed `task_contract.budget_profile`, inherited by subagents). Review gains a DISSENT layer: a minority reviewer with a concrete recommendation adds one compact non-veto capsule bullet (`acceptance_decision.dissent_noted`), ending silently-discarded correct minority findings. Under `required`+`blocking`, critical contributing findings become typed per-task `acceptance_obligations`; clean finalization asks for a per-obligation disposition through the extended `task_acceptance_review` tool, exhausted gates finalize honestly as `best_effort_open_obligations`, and every forced-finalization escape hatch bypasses the gate. `verify_and_record` records `criterion_source` (task_stated | agent_defined, default agent_defined) plus an optional `criterion_basis`, projected into the verification ledger and the reviewer's summary, with a one-shot advisory nudge for a basis-less agent-defined green. The loop latches an opt-in `CANDIDATES:` block beside FINAL ANSWER for reviewer adjudication (SYSTEM.md protocol), `web_search` results carry `answer_type=summary` with an open-the-sources doctrine line, and vision tool descriptions steer clean-frame extraction and native `view_image` over delegated/screenshot paths. |
Older releases are preserved in Git tags and GitHub releases. Older 6.x rows (including 6.56.0, 6.55.0, 6.54.2, 6.54.1, 6.54.0, 6.53.4, 6.53.0 and 6.51.0), the 5.2.0 through 5.33.0-rc.6 rows, and former `4.0.0` rows are rolled off to respect the P9 changelog cap; their full bodies remain at their git tags.

---

## License

[MIT License](LICENSE)

Created by [Anton Razzhigaev](https://t.me/abstractDL) & Andrew Kaznacheev
