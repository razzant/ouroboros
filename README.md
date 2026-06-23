# Ouroboros

[![GitHub stars](https://img.shields.io/github/stars/razzant/ouroboros?style=flat&logo=github)](https://github.com/razzant/ouroboros/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![macOS 12+](https://img.shields.io/badge/macOS-12%2B-black.svg)](https://github.com/razzant/ouroboros/releases)
[![Linux](https://img.shields.io/badge/Linux-x86__64-orange.svg)](https://github.com/razzant/ouroboros/releases)
[![Windows](https://img.shields.io/badge/Windows-x64-blue.svg)](https://github.com/razzant/ouroboros/releases)
[![OuroborosHub](https://img.shields.io/badge/OuroborosHub-skills%20marketplace-8A2BE2.svg)](https://github.com/razzant/OuroborosHub)
[![Version 6.42.0](https://img.shields.io/badge/version-6.42.0-green.svg)](VERSION)

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
| Heavy | empty → Main | Strong acting/coding lane (`OUROBOROS_MODEL_HEAVY`; renamed from `Code`, empty falls back to Main) |
| Light | empty → Main | Safety checks and fast helper tasks (`OUROBOROS_MODEL_LIGHT`, empty falls back to Main) |
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
| 6.42.0 | 2026-06-23 | **feat: faithful-benchmark declarative tool-policy (`disabled_tools`), agent deadline awareness, and native local image viewing.** Tools: an additive `task_contract.disabled_tools` tool-policy withholds named tools (hidden from `schemas()`/`get_schema_by_name`/`available_tools`, blocked at `execute`, inherited by subagents via the parent-contract spread) INDEPENDENT of `allowed_resources` — so a benchmark adapter can disable the agent's OWN web/search/VLM tools while leaving shell network egress (git clone/pip) intact, without tripping the web↔network cross-implication in the resource gate; threaded through `POST /api/tasks` and the frozen `TaskCreateRequest`. Deadline: the Terminal-Bench adapter resolves each task's Harbor wall-clock cap (from the cached `task.toml`) and hands it to the agent as `deadline_at` (minus elapsed install/server time + a safety buffer) so the one mind PACES itself (the 50/25/10% TIME-BUDGET milestones) and self-finalizes a best-effort partial result BEFORE Harbor's hard external kill instead of dying empty; spawned subagents now inherit the parent deadline into `task_metadata` (not just the contract) so their pacing/graceful-finalize fire too. Vision: a native `view_image` tool injects a LOCAL image file into the active model's own context (registered OUTSIDE `_WEB_TOOLS`, so it stays available under web-off). Benchmark: an honest disclosure-ledger taxonomy (pass / provider_infra / wall_clock_cancellation / genuine) that separates real wrong answers from provider/timeout/teardown artifacts. New surface: `task_contract.disabled_tools` + `TaskCreateRequest.disabled_tools`, `view_image`. |
| 6.41.0 | 2026-06-23 | **feat: safe merge-aware updates, project naming for non-human tasks, uniform full-output bubbles, an Activity subtab, and a skill-review token budget.** Updates (P2): the managed update now does a REAL git 3-way merge in an isolated temp worktree so local advanced/pro changes survive — `supervisor/update_merge_policy.py` classifies conflicts (clean / doc_reconcile / conflicting; docs auto-reconcile except BIBLE/CHECKLISTS/SAFETY), `supervisor/update_merge.plan_managed_update_merge` previews it, and a staged `POST /api/update/preflight` + `apply{strategy=auto_merge\|assisted\|manual\|replace}` lands a clean merge behind a fail-closed lock with a pre-restart smoke + transactional rollback + a post-boot boot-loop guard, while conflicts spawn a REVIEWED Ouroboros merge task (triad/scope). A main-screen Update pill + staged dialog surface it; the availability check runs on restart. Naming (P1): skill/system tasks (`skill_lifecycle_*`) with no human text now get a skill-derived name instead of the dead-end "New project", with a durable `project_named` reason-code event. Bubbles (P3): any truncated subagent/research bubble carries `{truncated, full_ref}` and expands inline to the genuinely-full output, fetched on demand into a bounded-scroll box. Activity (P4): a new Dashboard subtab shows cron schedules, running/queued tasks, and background consciousness with direct cancel/enable-disable controls (skill schedules read-only). Skills (P5): the byte-cap skill-review gate becomes a pack-level token budget (a 76 KB data file no longer locks a legitimate skill) with a chunked over-budget fallback, plus a first-class `write_surface='read_only'` subagent surface. |
| 6.40.0 | 2026-06-21 | **feat(core): LLM-first project naming, per-model self-DoS guard, soft join-ledger for subagents, orchestrator read-only roots, and turn-into-project ordering.** Naming: an SSOT `ouroboros/project_naming.py` coins a short human project title with a bounded LIGHT-model call (P5, fail-soft to a heuristic; no keyword gates); a proactive card namer (`project_naming.spawn_proactive_namer`) names a fresh main-chat card up front (background thread → `suggested_name` on the result + a `task_named` broadcast → the live card shows the title, persisted across reload via same-status enrichment), and **turn into project** reuses that name with NO extra call when it is ready — falling back to a bounded LIGHT `llm_project_name_async` only if the click beats the namer (the heuristic `task-…` default is gone). Resilience: a per-(model,route) `threading.BoundedSemaphore` (`ouroboros/model_concurrency.py`, `OUROBOROS_MODEL_MAX_CONCURRENCY`, default-on, fail-soft, deadline-bounded) caps concurrent provider calls WITHIN a process (a task's main loop + its in-process subagent threads + status pings) so they can't self-DoS one rate limit — PER-PROCESS like the fallback cooldown (a cross-worker governor is future work). Soft join (#7): `cancel_task` gains a recorded reason, a new `peek_task` inspects a child's status/beacons/result-tail WITHOUT absorbing it, a new `discard_child_result(task_id, reason)` is the EXPLICIT (not prose-parsed) signal to finalize without a child — both stamp a durable `parent_decision` the pre-finalization reminder honors — and a forced/deadline/provider finalization records any orphaned children instead of dropping them silently. Turn-into-project: the owner's request now sorts to the TOP of the project thread via a deterministic earliest-timestamp precedence (original chat-log send time → queued_at → result ts). Tooling: orchestrator-only READ-ONLY `subagent_projects`/`deliverables` resource roots (never write/shell, never to subagents); a freshly provisioned genesis project must be empty (fail-loud); a genesis project emits a typed `deliverable_manifest` on the artifact axis; a shared `_str_match_replace` gives the data-plane editor the same match feedback as the repo editor; and a data-plane shrink-guard blocks accidental overwrite truncation (force bypass). New surface: `peek_task` / `discard_child_result` tools, `OUROBOROS_MODEL_MAX_CONCURRENCY`, `subagent_projects` / `deliverables` resource roots. |
| 6.39.1 | 2026-06-20 | **feat(ui): complete the Phase-5 skill-trust UI that v6.39.0 shipped only the backend for.** Skills: an owner-only **⚠️ Skip review** action on the owner's OWN external/self-authored skill card POSTs the existing `/api/owner/skills/{skill}/attest-review` endpoint (danger-confirm; offered only while a review is outstanding or the attestation went stale, never once freshly attested), and an owner-attested verdict shows a distinct warning-toned **owner-attested** badge (the gateway now surfaces `review_profile`). Chat: a subagent live card shows a compact **role · model** label (provider prefix dropped, local route marked) — e.g. `planning-scout · gemini-3.5-flash` — preserved across the child's lifecycle even through model-less terminal events; and in a narrow chat column (project panel / mobile) the live-card title wraps to its own full-width line and the subagent nesting indent is trimmed so deep trees keep usable width. Frontend-only (plus a read-only `review_profile` gateway field); browser-verified by new Playwright `ui_browser` smoke tests. |
| 6.39.0 | 2026-06-20 | **feat(core): provider/FM correctness, model-slot role model with 429-resilient fallback, durable swarm coordination, atomic writes + browser visual verification, and owner-attested skills.** Provider boundary: a top-level `reasoning_content` is stripped outbound and on normalize (strict vLLM/SGLang no longer `400`s on the replayed field), a provider content-filter `400` writes the provider code + response BODY into the durable terminal event, cloud.ru model costs come from the live `/v1/models` catalog (RUB→USD, `is_billable`-aware, TTL-cached) instead of `$0` and the static table shrinks to providers without a pricing API, MCP servers are documented as first-class UNTRUSTED tools with a capability-omission surface when an enabled server yields zero tools, and **Max** context mode lazily PROBES the active route on first use (single-flight, fail-closed, observable "effective vs preferred") instead of silently downgrading to Low. Model slots: the `code` slot becomes **HEAVY** (default empty→Main, like consciousness/light), a comma `OUROBOROS_MODEL_FALLBACKS` chain replaces the singular fallback with a per-process 429-aware **cooldown** (`ouroboros/fallback_cooldown.py`, default-on, fail-soft, benchmark no-op) and cross-family reasoning sanitize, and an `auto` mutating child routes to HEAVY / a read-only child to LIGHT under a configurable depth cap with a visible downgrade note. Swarm: terminal/milestone attention beacons + contracts mirror from the ephemeral task-tree ledger into the DURABLE project journal, an unknown journal kind is recorded as a loud `note`, and child artifact pointers are STAT'd as ground-truth facts. Tooling: a `write_text_atomic` SSOT makes every overwrite crash-safe (a crash mid-write leaves the OLD file intact), and the bundled browser launches with SwiftShader so WebGL renders (not black), waits for paint (`readyState` + double-rAF), and wraps a bare-`return` evaluate in an IIFE; `vlm_query` reads local `file://` images via the base64 path. Skills: the owner can SKIP only the expensive LLM review for their OWN external/self-authored skill via an owner-only `POST /api/owner/skills/{skill}/attest-review` that STILL runs the full deterministic preflight + `SkillManifest.validate()` floor, binds a content-hash verdict to an agent-unforgeable owner-state marker (invalidated if the marker is removed), refuses third-party/marketplace/native sources, and is barred from public-hub publication and agent self-call. New surface: `POST /api/owner/skills/{skill}/attest-review`, `OUROBOROS_MODEL_HEAVY` / `OUROBOROS_MODEL_FALLBACKS`, `ouroboros/fallback_cooldown.py`, `ouroboros/skill_owner_attestation.py`. |
| 6.38.1 | 2026-06-19 | **fix: complete three v6.38.0 invariants a cumulative-with-plan review surfaced (the per-commit reviewers never saw the plan).** Reaper: when a timed-out worker will not confirm dead, the off-loop reaper now does NOTHING that could race it — no terminal write, `task_done`, retry, or respawn — holds the slot `reaping`, and persists a durable `STATUS_RUNNING` result (rank-2, dropped by the monotonic merge guard if the worker self-finalized) so the orphan is reconciled to `failed` on the next generation instead of vanishing into limbo; a `task_reaper_wedged` event + an owner `/restart` hint surface it. Swarm: the `interface_contract` child→parent beacon (raised when the shared seam/contract must change) is now a real attention kind that early-returns a parent's `wait`, named consistently across the `tree_note` schema, the `SYSTEM.md` doctrine, the injected ledger header, and the `wait_task`/`wait_tasks` descriptions; the task-tree ledger validates `root_task_id` strictly (`validate_task_id`) instead of silently sanitizing. Skills: the explicit `runtime_data` `skills/<bucket>/<skill>/…` write path now applies the SAME manifest-first typo guard as the bucket/skill_name short-form (shared `is_skill_create_typo`, run on the normalized path) so a misspelled name can no longer mkdir a bogus payload. Bugfix-only; no gateway-contract change. |
| 6.38.0 | 2026-06-19 | **feat: honest non-blocking orchestration, provider-agnostic narration, light skill creation, and Swarm/Deliverables UX.** Orchestration: every turn's context carries a `capabilities` digest (`allow_mutative_subagents` from the live MASTER gate — light blocks only self-repo/control-plane, NOT user deliverables or acting children) and a live `queue` digest, both from the same getters the runtime enforces; the flat wall-clock kill is replaced by an ACTIVITY model (stop only on no REAL progress AND no progressing/queued subtree past an idle window floored to the per-call ceiling; the only HARD axes are an explicit `deadline_at`, a 6h ceiling, and budget; an orchestrator with live children is never blind-retried); the heavy worker teardown moves OFF the supervisor loop to a single-owner reaper (`supervisor/task_reaper.py`, slot marked `reaping` under `_queue_lock`); and a domain-agnostic task-tree ledger (`tree_note`/`tree_read`, `data/task_trees/<root>/blackboard.jsonl`) is the swarm blackboard + typed child→parent beacons that early-return a parent's `wait`. Narration: an empty tool-round bubble now surfaces readable reasoning the provider already returned (`LLMClient.extract_display_reasoning`, shape-based, opaque skipped, DISPLAY-ONLY, `OUROBOROS_REASONING_SUMMARY`). Skills: light can again CREATE a new external skill from scratch by writing its `SKILL.md`/`skill.json` manifest, with an advisory `capability_regression` review item + golden from-zero tests. UX: the Consilium pill becomes **Swarm** (deep plan via `plan_task` THEN subagent fan-out; the `force_plan` contract is preserved, no gateway bump for behavior); a BARE `user_files` filename lands in the visible `~/Ouroboros/Deliverables/` container (`OUROBOROS_DELIVERABLES_ROOT`) instead of cluttering the home root; and a converted project thread shows the owner's request at the top (stamped with its original timestamp). |
| 6.37.1 | 2026-06-18 | **fix(lease): a UI-converted task holds its project's one-writer lease.** "Turn into project" (`api_project_from_task`) bound the task durably but never updated the supervisor's in-memory `RUNNING`/`PENDING` — the one-writer lease and assignment read `task['project_id']` from those live structures, NOT the durable bindings, so a converted task could let a concurrent same-project task be assigned (two writers), and a still-PENDING converted task would start unscoped and miss its lane. The convert path now updates both under the queue lock via a shared `mark_task_project(RUNNING, PENDING, …)` SSOT helper (generalized from the in-task `ensure_project_scope` path so the two cannot drift). Pre-existing gap surfaced during the v6.37.0 review; bugfix-only, no contract change. |
| 6.36.2 | 2026-06-17 | **feat(process-custody): reap skill-companion orphans (log-only).** `reap_orphaned_processes` gains a companion-aware branch: a skill companion (daemon scope, `purpose companion:<skill>:<name>`) is reaped when its owning skill is uninstalled OR the ledger entry is from a foreign (dead) server generation — closing the gap where a stale companion left by a previous generation lingered and could block re-spawn on a port conflict (in-process `stop_skill`/`panic_kill_all` only cover the live generation). Ships **log-only** (`enforce_companion_reap=False` emits a `process_would_reap` event instead of killing); **fail-safe** — an unknown or momentarily-empty installed-skill set coalesces to keep-all (never a mass-kill), strict `(pid, start_time, cmd_sha256)` fingerprint, and same-session companions of installed skills are always kept so the live `CompanionSupervisor` stays their sole owner. `server.py` derives the installed-skill set from disk for the reaper. |
| 6.36.1 | 2026-06-17 | **fix(build): Windows release build — tolerate `compileall` per-file failures.** The v6.36.0 macOS-signing bytecode-seal step (`compileall`) returns a non-zero exit when any single bundled file fails to compile (the `python-standalone` ships a known tab/space-broken Tcl/Tix `WmDefault.py` that Ouroboros never imports). On macOS/Linux that was already tolerated via `|| true`; the Windows `.ps1` lacked the equivalent, so under pwsh 7.4+ (`$PSNativeCommandUseErrorActionPreference`) the native non-zero exit aborted the build. The Windows build now neutralizes Stop-on-native-error around `compileall` and resets `$LASTEXITCODE` (parity with the POSIX `|| true`) — the rest of the tree is still sealed for start-speed/signature parity. macOS/Linux builds were unaffected and already green. |
Older releases are preserved in Git tags and GitHub releases. Older 6.x rows, the 5.2.0 through 5.33.0-rc.6 rows, and former `4.0.0` rows are rolled off to respect the P9 changelog cap; their full bodies remain at their git tags.

---

## License

[MIT License](LICENSE)

Created by [Anton Razzhigaev](https://t.me/abstractDL) & Andrew Kaznacheev
