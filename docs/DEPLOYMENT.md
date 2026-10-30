# DEPLOYMENT.md — Deployment Notes

## Roots, Ports and One Server per Data Directory

Paths: `OUROBOROS_REPO_DIR` (checkout), `OUROBOROS_DATA_DIR` (memory/state/settings/logs),
`OUROBOROS_SETTINGS_PATH`, `OUROBOROS_PID_FILE` (launcher lock), `OUROBOROS_PORT_FILE`.
Checkout, data root and PID file default beneath `OUROBOROS_APP_ROOT` (`~/Ouroboros`);
the settings and port files default beneath the data root, so an explicit DATA_DIR
supplies their defaults too. No root is inferred from another's parent. Work roots
(`OUROBOROS_SUBAGENT_PROJECTS_ROOT`, `OUROBOROS_SUBAGENT_WORKTREE_ROOT`,
`OUROBOROS_DELIVERABLES_ROOT`) prefer saved settings,
then environment, then the data root's basis (`state/state.initialized.json`):
`<APP_ROOT>/<folder>` for data first initialized by this release or later,
`~/Ouroboros/<folder>` for older data. An unreadable basis makes a default work root
refuse until its key is set; nothing is moved, copied or deleted. Compose/systemd/shell
may load an env file; Ouroboros does not. A direct server uses its checkout and
`<APP_ROOT>/data`; while DATA_DIR is unset and `~/Ouroboros/data` holds memory (or cannot
be examined), it exits 2 naming both choices: that data with the old
`<APP_ROOT>/data/settings.json`, or `<APP_ROOT>/data` (ARCHITECTURE §7).

`OUROBOROS_SERVER_PORT` defaults to 8765; fallback skips Host Service/local-model
ports. Source CLI, desktop CLI, Settings, extensions and launcher use the published
port of the roots they resolve. Desktop `run --start` accepts a running server only when
that data root's service binding names it (`--url` names one outright); otherwise it opens
the app, on macOS through `open -n --env` with its resolved paths, including defaults. Keep containers'
published port free.

An older release ignores the basis: it creates new work roots under `~/Ouroboros`, and
its direct server given only APP_ROOT keeps memory in `~/Ouroboros/data`. Set DATA_DIR
and the three work-root keys explicitly before running one on such data.

One server holds `<data>/state/server.lock` until exit, regardless of repository,
PID file or port. A second waits briefly for Restart, then exits 2;
unsupported kernel locks refuse startup. Stop releases predating this lock first.
`state/install.json` binds the root. Boot/write atomically mints and logs a
copy/move identity, clearing cached verdicts. Checks cache paths; identity stat
changes trigger parsing. Stop before copy/move. Paths never authorize
foreign locks, transactions or processes. Layout children belong to their install;
bare roots work. Foreign tasks keep local history and rest with
`foreign_install_root` until explicitly moved.

To move a task: stop and back up both installs, copy its required child files
into the destination's task layout, then call
`task_results.write_task_result` on the destination with the existing status,
updated `budget_drive_root`, `drive_root`/`child_drive_root` as applicable, and
cleared `reason_code`/`reason_detail`. Keep sources through adoption.
Copying the directory alone never rewrites those pointers.

## Unknown model role at startup

An error such as `OUROBOROS_MODEL_CONTEXT_WINDOWS: unknown model role(s)`
means the code reading that setting does not recognize a role name. Current
diagnostics name the setting and offending roles, without dumping their values.
Older packages print `OUROBOROS_MODEL_CONTEXT_WINDOWS must contain only known
model roles` without identifying the role; the same recovery below applies.
Check for a misspelling or a version mismatch. The same strict rule applies to account pins
and processing preferences: unknown entries are not silently ignored or changed
to Auto, even when a context window is zero.

The packaged launcher and the managed repository are separate. Updating the
repository, including a manual `git pull` in a fork, does not replace an older
executable. Current launchers can hand off to updated source after an import
probe, but an older executable without that handoff still needs an application
package update. New source cannot repair its frozen settings parser or logging.

To recover without resetting the installation:

1. Identify the executable you actually launch, its package/build, the user it
   runs as, and the effective paths listed above. The default native Linux
   package is `/opt/ouroboros/Ouroboros`; AppImages and tarballs live wherever
   you placed them. A repository version alone does not identify the launcher.
2. Stop that instance through its normal launch path. If you chose the optional
   systemd unit, see its [stop and log instructions](../packaging/systemd/README.md)
   and shared-daemon caveat. Back up the actual data directory, settings file
   (if separate), and repository/local changes before changing anything.
   Do not delete `settings.json`, memory, or the checkout to get past this error.
3. If the name is genuinely misspelled, correct only that entry after checking
   its intended role. If another version wrote a legitimate role, identify which
   reader is stale from the traceback. A frozen launcher needs a compatible
   package: upgrade the `.deb`/`.rpm`, or replace the AppImage/tarball application
   files; keep user data and the managed checkout. If the managed checkout is
   older (for example, after a rollback or in a pinned fork), use a compatible
   checkout while preserving local changes. A newer package alone will not fix
   that reader. Do not strip an unfamiliar role merely to bypass validation.
   Default native package files under `/opt/ouroboros` are separate from
   `~/Ouroboros/data` and `~/Ouroboros/repo`. Verify any custom layout first.
4. Launch as the same user with the same path overrides, then check the saved
   model/account selections and logs. Normal bootstrap/migrations still run;
   package replacement is not a promise that every file remains byte-identical.

Current launchers log uncaught main/thread exceptions. With an older package,
`launcher.log` can contain only a crash-fuse summary while the actual traceback
went to stderr. After stopping the old instance, run the same executable from a
terminal and retain its output, or read the unit journal if that was the launch
path. Default logs are in `~/Ouroboros/data/logs/`; use the actual data root if
overridden. Share only the relevant traceback and package identity, never the
whole settings file or credentials. Repeated restarts do not fix a parser mismatch.

## Containers

Two images, one recipe file: [`packaging/docker/compose.yaml`](../packaging/docker/compose.yaml)
is the executable recipe for both, built from the repository root. Run one
profile per volume set; never point two containers at one data volume.

| | Immutable image (`ouroboros-web`) | Managed body (`ouroboros-managed`) |
|---|---|---|
| Code and packages | the image's `/app` | a checkout and runtime venv on a volume |
| Update | build and deploy a new image | Settings → Update, as on the desktop |
| Entry | `python server.py` | the common launcher, `--no-ui` |
| Volumes | data, work, `HOME` | one parent volume, or body/data/work/home |

**Interpreter.** The image's Python is python-build-standalone release 20261009:
CPython 3.10.22 and SQLite 3.53.1, the same assets
`scripts/download_python_standalone.sh` and `scripts/download_python_standalone.ps1`
pin for the desktop packages. It is not Debian's `python:3.10-slim` (SQLite 3.46.1).
Desktop installs receive that interpreter only by installing the app again.
Settings → Update replaces the repository and its packages; it does not replace
the bundled interpreter. A managed container keeps its runtime venv on a volume.
A venv whose interpreter no longer matches this image is left untouched, and the
log names the recovery step: stop, move that directory aside, start again. Code
and data stay; packages that lived only in the old venv do not.

**Exposure.** Both images bind `0.0.0.0` and set no `OUROBOROS_NETWORK_PASSWORD`:
a missing password only warns and the gate then admits every request, so a
published port without it exposes the whole owner surface. The recipes publish on
`127.0.0.1`; set the password before publishing further, or keep the port
unpublished. Packaging adds no stronger boundary of its own.

**Managed body.** `docker build --target managed --build-context
ouroboros-seed=<dir> .` takes a released seed, the `repo.bundle` and
`repo_bundle_manifest.json` that `scripts/build_repo_bundle.py` emits from a clean
annotated release tag; the build refuses a bundle without complete history. The
image keeps only the launcher; its layout puts the body at
`/ouroboros/body/{repo,venv}`, data at `/ouroboros/data`, work at
`/ouroboros/work` and `HOME` at `/ouroboros/home`. First boot clones the seed into
the body and prepares `--runtime-venv` (`ouroboros/runtime_venv.py`): it needs
network for the dependency lock, and an interrupted preparation resumes on the
next start. Later boots, Update, self-modification and Restart run on the volume;
removing and recreating the container keeps body, packages, data, work and
`HOME`. Mount volumes at `/ouroboros` or at the four kind directories, never at
the repository itself: bootstrap refuses a mount point without a checkout and
moves nothing. At exit 42, changed launcher helpers imported from `/app` hand off to the volume
checkout, as at cold start; server-only changes keep the shell. A new image supplies
the startup shell and seed for future first boots; a new Python, system package or browser build still needs a new image, and
a venv created by another Python is refused untouched with the recovery step
(move it aside; the next start prepares a fresh one). Git rollback restores code
and its lock, not data or the exact package set.

**Restart policy and Panic.** Use `restart: on-failure` with automatic intent
(both recipes): after Panic the automatic entry exits 0, leaving the stop marker
intact, so the container stays stopped; Panic itself exits 99 in a direct server
and 0 in the launcher. `always` and `unless-stopped` keep restarting clean exits;
without automatic intent a restart policy consumes the marker. The recipes show
the foreground owner resume (`--launch-intent owner`, or
`OUROBOROS_LAUNCH_INTENT=owner` for the immutable image); an ordinary `docker
start` keeps automatic intent, and deleting the marker is never the resume. A
refused start (a held data directory, a mounted repository, a foreign venv) exits
nonzero, so `on-failure` retries it with Docker's growing delay until the cause is
fixed; a second managed container sharing the data volume loses the launcher lock
and exits 0. `init: true` reaps orphans the launcher or server cannot. See
Docker's [restart policy documentation](https://docs.docker.com/engine/containers/start-containers-automatically/).

**Boot health.** Each recipe probes `GET /api/health` inside the container. The
probe succeeds while the process is alive and `boot.phase` is not `failed`,
including before the server opens its socket. There is no start deadline: a long
boot does not become an unhealthy container and does not by itself restart it.
`restart: on-failure` still follows a real process exit. `boot.phase` `failed`
marks the container unhealthy.

## Trusted Docker / Kubernetes Non-Local Binds

By default, saving `OUROBOROS_SERVER_HOST=0.0.0.0` through the Settings UI
requires `OUROBOROS_NETWORK_PASSWORD` in the same save. This keeps desktop and
local-network launches from accidentally exposing the full Ouroboros HTTP and
WebSocket surface without the built-in password gate.

Trusted container deployments may opt out with:

```bash
OUROBOROS_TRUST_NONLOCAL_BIND_WITHOUT_PASSWORD=1
```

Use this flag only when access is already restricted by external
infrastructure, for example:

- ingress authentication
- VPN-only routing
- private Kubernetes service/network policy
- an authenticated reverse proxy

With the flag enabled, Ouroboros still warns when saving a non-localhost bind
without `OUROBOROS_NETWORK_PASSWORD`, but the Settings UI no longer blocks
ordinary settings saves such as API-key updates. Do not use this flag on an
open LAN or public port.

## The Main Interface and a Separately Hosted One

`OUROBOROS_UI_DIR` chooses what the core serves at `/` and `/static/*`:
empty for the builtin interface, `none` for no main page, or an absolute (or `~`)
directory with its own `index.html`, laid out like `web/`. It applies when the
server starts, so a saved change waits for a restart. A saved value, an empty one
included, beats the environment variable; the environment counts only while the
setting was never saved, and Ouroboros never copies it into `settings.json`. To
return to the builtin interface, clear the setting (in Settings → Advanced, or in
`settings.json` while Ouroboros is stopped), restart and reopen the address.
`GET /api/state` reports what is actually served as `served_ui`.

- A directory is trusted, not sandboxed: every file in it, dot-files included, is
  published under `/static/`, so point it at a dedicated build output. Its HTML is
  served unchanged; a custom interface used as a Telegram Mini App includes
  Telegram's script itself.
- A directory that is relative, missing, has no `index.html` or contains the
  core's data, source or settings tree is not applied: the builtin interface
  serves and `served_ui.error` says why.
- A selected directory may also supply complete `onboarding.html` and
  `auth/index.html` pages. The login's `auth/` assets are public before sign-in;
  publish only that page's public build files there. Missing optional pages
  fall back whole to builtin. `/onboarding?builtin=1` and
  `/auth/login?builtin=1` keep builtin recovery without changing the setting.
  API writers and password checks remain the core's (§2–4 of CLIENT_CONTRACT).
- With `none`, `/` answers a 404 page and `/static/*` 404; windows that open the
  core's own address (desktop, Android, Telegram) show that page too. The API,
  the WebSocket, sign-in, the setup wizard (`/onboarding`, with its own assets
  under `/onboarding/static/`), extension routes and files keep working.

An interface hosted elsewhere must share one browser origin with the core:
cross-origin access and mounting under a sub-path are not supported. A reverse
proxy sends the core's prefixes (`core_paths` in `GET /api/contract`: `/api`,
`/auth`, `/onboarding`, `/ws`) to the core and everything else to the interface
host, passes WebSocket upgrades, keeps request bodies and event streams
unbuffered and never answers a core path with the interface's page. Client rules
are in [`CLIENT_CONTRACT.md`](CLIENT_CONTRACT.md); a verified Caddy setup with
its coverage limits is in [`examples/separate_ui/`](examples/separate_ui/).

The password gate asks only non-loopback clients, and behind a proxy the client
is what the proxy says. uvicorn takes `X-Forwarded-For` and `X-Forwarded-Proto`
only from the addresses in `FORWARDED_ALLOW_IPS` (default `127.0.0.1`), using
the last `X-Forwarded-For` entry that is not such a proxy:

- A proxy on the core's machine must set `X-Forwarded-For` to the peer it saw.
  Without the header every visitor reaches the core as loopback, and a proxy that
  passes a visitor's own header through unchanged lets that visitor claim
  loopback; either way the password is skipped. A browser on the proxy's machine
  is loopback by design.
- A proxy on another machine must be listed in `FORWARDED_ALLOW_IPS`; otherwise
  the core treats the proxy itself as the client and does not see TLS, so the
  session cookie is not marked `Secure`.
- The core accepts a browser's requests and WebSocket handshakes by the
  browser's `Sec-Fetch-Site: same-origin`, so the proxy may rewrite `Host` or
  terminate TLS. A browser that sends no Fetch Metadata is accepted only when
  its `Origin` equals the scheme and host the core sees: pass `Host` through,
  plus `X-Forwarded-Proto` from a proxy in `FORWARDED_ALLOW_IPS`.
- Visitors who are not loopback need `OUROBOROS_FILE_BROWSER_DEFAULT` for Files;
  without it the path-addressed `/api/files/*` operations refuse them.

With or without a password, on loopback too, the core answers another site's
page with 403 (a WebSocket closes with 4403) and a body that is not JSON where
the API expects JSON with 415. Pages and files any page may load (the interface,
setup and login pages, their static files, extensions' reviewed widget files)
pass this check; the password gate above still applies. A request with neither
`Origin` nor `Sec-Fetch-Site` counts as a native client (the CLI, sidecars,
skills). Browsers send Fetch Metadata only to HTTPS, `localhost` and loopback
addresses and omit `Origin` on simple GET/HEAD loads, so through a plain-HTTP
hostname another site's page can still make the browser run GET/HEAD handlers,
without reading the answers. A hostname that resolves to this machine (DNS
rebinding) is not told apart either.

## Extra CA Certificates

Ouroboros verifies its own provider calls (OpenRouter, OpenAI-compatible endpoints,
Anthropic, GigaChat, model catalogs and Provider Test) against the `certifi`
bundle, never the operating-system store. A deployment behind a TLS-inspecting
proxy, or one that talks to an endpoint signed by a CA `certifi` lacks (for
example the Russian Trusted Root CA behind GigaChat), sets
`OUROBOROS_EXTRA_CA_BUNDLE` to a PEM file holding the missing CA certificates.
The file is added on top of the defaults, so every other provider keeps working;
a path that cannot be read fails the call loudly instead of silently falling
back to the defaults. In Docker, mount the file and pass the setting:

```bash
docker run --rm -p 8765:8765 \
  -v "$PWD/extra-ca.pem:/certs/extra-ca.pem:ro" \
  -e OUROBOROS_EXTRA_CA_BUNDLE=/certs/extra-ca.pem \
  ouroboros-web
```

On a desktop install the same key lives in Settings → Advanced. It covers the
provider calls: model requests, model catalogs, Provider Test, pricing and
capability probes. Everything else keeps its own trust store: `git`, `uv`,
`pip`, the Claudexor engine and its runtime downloads, the Telegram skill, MCP
servers, the web-search scraper and the browsers.

## Logs and External Monitoring

Ouroboros sends no telemetry; nothing below leaves the machine until the
deployment connects a destination of its own.

- The server process writes `logs/server.log` (2 MB × 4) and its stderr. Each
  pool worker logs to its stderr only: the desktop launcher copies it into
  `logs/agent_stdout.log`, Docker keeps it as the container log.
- The server and every pool worker configure this logging at their real
  start (`ouroboros/process_logging.py`), whatever the entry point (`python
  server.py`, `ouroboros server`, Colab); the desktop launcher keeps its own
  `logs/launcher.log` and routes its uncaught exceptions there. Under
  `OUROBOROS_WORKER_START_METHOD=fork` a worker inherits the server's handlers
  instead (`docs/PERSISTENCE.md`). Not covered: whatever runs before that call
  (the entry module's own imports), out-of-process extension children, the
  startup historical-audit child, the local-model server and other helper
  processes. An unexpected failure — a task
  exception, a worker crash, an unhandled gateway request (HTTP 500), an
  uncaught thread exception — is a stdlib `logging` record with its traceback,
  so a handler attached in that process receives it. Message text passes the
  secret-redacting filter; tracebacks and structured extras do not.
- File collectors (OpenTelemetry Collector, Vector, Fluent Bit) can ship durable
  JSONL under `logs/` and task drives without changing Ouroboros (`docs/PERSISTENCE.md`).
  The record passport, `ouroboros/contracts/record_contract.py`, defines guaranteed
  anchors/fields, correlation, content, rotation, tool and model-error replicas, child-drive retention
  (`OUROBOROS_GC_RETENTION_DAYS`) and money's source: accounting views, never summed
  `llm_usage`. Its `LEGACY_RULE` explains v2 nullable tokens: retain the stamp and nulls;
  old zeros stay ambiguous. `docs/examples/log_collector/` is a verified metadata-only
  Vector setup.
- An error tracker attaches from inside the process: an in-process extension
  skill loads in the server and in every worker, and can initialise a
  Sentry-compatible SDK (self-hosted Sentry, GlitchTip) installed into
  Ouroboros's own environment (in Docker, a layer of the image; a skill-declared
  dependency would move the extension out of process, where it cannot see these
  records). Turn frame-local capture off and pass events through
  `ouroboros.observability.redact_projection` before they leave;
  `docs/examples/error_tracker/` is a verified extension that does this and
  says what still leaves (messages and exception text can quote task text).
  Failures that happen before the extension loads stay in the local logs only,
  as do the launcher's and the Claudexor daemon's.
