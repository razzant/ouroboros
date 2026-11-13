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
may load an env file; Ouroboros does not. A direct server uses its checkout (Restart keeps it as is, or
refuses before the stop if dependencies or the import fail) and
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

## First Run Without the Wizard

The setup wizard opens while no model access is configured. A server that finds
access in its environment (a provider key, a subscription Main or a local route)
starts working at once and does not open it; the wizard stays reachable at
`/onboarding`. The check is that access is configured, not that it works: a wrong
key shows at the first request. That is the usual first run of a container or a
service unit configured by variables alone, and of a Colab notebook, whose launch
writes the whole factory settings document with the budget and the Access level it
was given. No wizard choice is saved then, and what the wizard would have asked
keeps its factory value:

| | Factory value | After completing the wizard on fresh data |
|---|---|---|
| Safety coverage | `full`: commands are read first too | `light`: outgoing actions are read first (background consciousness keeps `full`) |
| Access level | `advanced` | the level chosen |
| Limits | $200 in total, $50 for one task and its helpers, of known spend | the limits entered |
| Consciousness | off; $20 over 24 hours once started | off unless ticked |

Reviewers and helpers are proposed from the reachable Main at each read (none while
the keys present do not settle which provider serves Main) until they are edited on
the Agents page or saved by the wizard; other saves, the Models page's among them,
leave them proposed. A Colab launch saves the ones it proposes from the key it was
given. Subscription sessions join them through Settings or the wizard.

A setting the file does not hold is read from the environment variable of its name
(a Colab launch writes the ones named here into its file: `TOTAL_BUDGET` and
`OUROBOROS_RUNTIME_MODE` from the notebook's variables at each launch that sets them,
the rest as the file held them, or at factory values on a first launch), for example
`TOTAL_BUDGET`, `OUROBOROS_PER_TASK_COST_USD`, `OUROBOROS_CONSCIOUSNESS_DAILY_USD` and
`OUROBOROS_RUNTIME_MODE` (read at a start). The exceptions are the choices only the
owner's controls author, Safety coverage and the Context mode among them: Settings
shows their saved or factory value whatever the environment says. Neither is
consistent there today: while none is saved, the Safety Supervisor still reads a
forwarded `OUROBOROS_SAFETY_MODE`, and a task still starts in a forwarded
`OUROBOROS_CONTEXT_MODE` (#3202), so choose both in Settings to have one answer. A
deployment policy file (below) can cap the Access level and raise the minimum
coverage.

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
| Update | build and deploy a new image | Dashboard → Updates, as on the desktop |
| Entry | `python server.py` | the common launcher, `--no-ui` |
| Volumes | data, work, `HOME` | one parent volume, or body/data/work/home |

**Interpreter.** The image's Python is python-build-standalone release 20261009:
CPython 3.10.22 and SQLite 3.53.1, the same assets
`scripts/download_python_standalone.sh` and `scripts/download_python_standalone.ps1`
pin for the desktop packages. It is not Debian's `python:3.10-slim` (SQLite 3.46.1).
Desktop installs receive that interpreter only by installing the app again.
Dashboard → Updates replaces the repository and its packages; it does not replace
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
the startup shell and seed for future first boots. New Python or system dependencies
require a new image; WebKit follows the image too. The managed body's Chromium pin
can instead be downloaded into `/ms-playwright` on first use. A venv created by
another Python is refused untouched with the recovery step
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

**Fault exit.** A ready server whose event-bus process ended (`bus_lost`) or whose task
loop failed three times in a row (`loop_failed`) ends its generation through the ordinary
teardown (saved work held, the cause in `shutdown_entry.json`) and comes back with the
same code: under the launcher (the managed image, systemd, the desktop apps) it exits 1
and the launcher's crash path restarts it (fuse: five crashes in 120 s); a direct server
(the plain image, `ouroboros server`, Colab) re-executes itself in place after its own
process census, and only a failed successor exits 1, which `restart: on-failure` then
retries. After a fault boot the mind gets one recovery turn in Main with the facts (the held
tasks, the roots cancelled without a saved point): one paid Main turn per fault boot, so a fault
that returns after every boot costs one turn per cycle. No new turn starts when the server fell
while the previous recovery turn was going (admitted in the generation that fell, unfinished); an
earlier turn waiting with saved work does not block a later fault's turn. Neither path is a
planned restart (exit 42) or Panic (99); the shared Claudexor
daemon is untouched, while the kept tasks' own delegated runs end. A persistent cause cycles about every two minutes with a Main row
each time until someone looks.

**Browser sandbox.** Browser tools launch only the pinned Chromium (`ouroboros/chromium_runtime.py`)
inside its own sandbox. The images run as root, for whom Chromium refuses its sandbox (Docker's
default seccomp profile also restricts the user namespaces it uses), so by default Chromium is
refused with `BROWSER_CHROMIUM_SANDBOX_UNAVAILABLE`; WebKit stays available. An administrator who accepts
running web content without that sandbox sets `OUROBOROS_CHROMIUM_SANDBOX=off` in the container
environment: each launch logs a warning, and browser-condition annotations on page, screenshot
and evaluate observations disclose the opt-out. Unrecognized values are refused; boolean aliases are accepted. Nothing turns the sandbox off on its own.

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
return to the builtin interface, clear the setting (in Settings → This computer → Advanced, or in
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
  session cookie is not marked `Secure` and Chromium's WebSocket is refused.
- The proxy passes the browser's `Host` unchanged, port included, and its TLS as
  `X-Forwarded-Proto: https`; the core never reads `X-Forwarded-Host`. The
  browser's `Sec-Fetch-Site: same-origin` passes a request's origin check even
  with a rewritten `Host` (a malformed or `null` `Origin` is still refused, and
  the password gate still applies), but Chromium (151 and 157 tested; Firefox
  153 and WebKit 26.5 add the header) opens the WebSocket with `Origin` alone,
  accepted only when it equals the scheme and host the core sees. A proxy that
  breaks this serves a working page whose live updates never connect
  (`Reconnecting… · server answers`); the core logs each refused `origin=`
  beside the `expected=` origin it saw.
- Visitors who are not loopback need `OUROBOROS_FILE_BROWSER_DEFAULT` for Files;
  without it the path-addressed `/api/files/*` operations refuse them.

With or without a password, on loopback too, the core answers another site's
page with HTTP 403, including at the WebSocket handshake (browsers usually report
1006, not the server's internal 4403). Pages and files any page may load
(the interface, setup and login pages, their static files, extensions' reviewed
widget files) pass this check; the password gate above still applies. A request with neither
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
docker run --rm -p 127.0.0.1:8765:8765 \
  -v "$PWD/extra-ca.pem:/certs/extra-ca.pem:ro" \
  -e OUROBOROS_EXTRA_CA_BUNDLE=/certs/extra-ca.pem \
  ouroboros-web
```

On a desktop install the same key lives in Settings → This computer → Advanced. It covers the
provider calls: model requests, model catalogs, Provider Test, pricing and
capability probes. Everything else keeps its own trust store: `git`, `uv`,
`pip`, the Claudexor engine and its runtime downloads, the Telegram skill, MCP
servers, the web-search scraper and the browsers.

## Deployment Policy File

A deployment can fix choices its employees and their Ouroboros may not change. Its launch
(compose, systemd, the container runtime) names one strict JSON file in
`OUROBOROS_DEPLOYMENT_POLICY`: an absolute path outside the data root and the checkout,
which are refused. Without the variable nothing changes. A named file that is missing,
unreadable, malformed or of another version stops `server.py` with exit 2 before it serves,
and any other process that reads settings under it refuses: a bad file never reads as
unmanaged.

```json
{
  "version": 1,
  "access": {"max": "advanced"},
  "safety": {
    "min_coverage": "full",
    "checker": {"model": "anthropic@safety::<model>", "fallbacks": ["openai@safety::<model>"]},
    "allowed_exits": []
  },
  "rules": "Customer records stay inside the corporate network.",
  "extensions": {
    "when_unmet": "hold",
    "required": [{
      "name": "error_tracker",
      "path": "/ouroboros/data/skills/external/error_tracker",
      "content_hash": "<the 64-hex content_hash GET /api/extensions reports for it>",
      "settings": {"SENTRY_DSN": "https://<key>@errors.example.internal/<project>"}
    }]
  }
}
```

`version` (1) is required; every other field is optional and constrains only what it names.
An unknown or repeated field or a wrong type refuses the whole file. Safety is mandatory, at
full coverage, on the organization's checker only when the file names all three of
`access.max` at most `pro`, `safety.min_coverage: full` and `safety.checker`: without the cap an
employee may still choose Cyber Pro, where Safety findings are advice; without the floor
coverage may be lowered to Outgoing or Off; without `safety.checker` the employee's own
Supervisor route checks.

- `access.max` (`light`, `advanced` or `pro`) caps effective Access, keeping Cyber Pro out of reach.
- `safety.min_coverage` (`light` or `full`) floors both Supervisor coverage levels; the
  new-install Light default yields to it.
- `safety.checker` is the Supervisor's route: `model` (any model route), `account` (a
  subscription profile, for `claudexor::` routes only; empty rotates) and up to eight
  `fallbacks`, the only routes its walk tries, in order; a route whose connection lacks its
  credential is skipped unsent. Settings slots, role options, wait-card switches and
  second-check rows never move it. When none of its routes answers, the action that needs
  the check is held, typed `SAFETY_UNAVAILABLE`, until a check answers or the work is
  cancelled (unless `release_unjudged` is named); reads, Stop and Panic stay available. A save
  after which a checker route would reach another endpoint, key or local model server, judged as
  a send reads settings (defaults, the legacy compatible pair and launch variables included), is
  refused, so give the checker connections of its own; such a field carried back as shown keeps
  the document's own value. A local checker's server starts only with the model, template and
  port its saved settings name; Stop stays available.
- `safety.allowed_exits` lists the exits left open, each once; `[]` closes these three at every
  Access level. `repeat_after_dangerous`: the agent's identical call from a later step lets a
  returned DANGEROUS call of any tool through in Cyber Pro. `release_after_dangerous`: an employee's
  answered card holding the exact words and destination (any other call: the exact call) releases
  a returned DANGEROUS call, which outside that Cyber exit waits for it. `release_unjudged`: the
  same card releases a call without a verdict. A card may precede the send; it proves no
  assessment. Closed exits leave a returned call refused, or an unjudged call waiting for a check.
  Under every list, a later message or card answer of the employee in the same work earns a waiting
  DANGEROUS call one new reading on new evidence, not a release.
  Unretained verdicts still prevent unchecked release in the same context; losing all state
  requires a fresh check unless unchecked release is enabled. An explicit list never gains another
  exit implicitly. Omitted, the ordinary exits remain except `release_unjudged` under `safety.checker`,
  where naming it weakens mandatory checking. A SUSPICIOUS call passes on a later identical call.
- `rules` (at most 64,000 UTF-8 bytes) reach every acting mind, delegated session and Safety
  check as the organization's rules, beside the employee's own, which cannot relax them.
- `extensions.required` names up to 16 installed extensions that must run, each by `name` (its
  folder name), `path` (the package directory, absolute) and `content_hash`, the digest the loader
  computes for the files it runs; a source's or release's digest is provenance, not this pin.
  Exactly that package reads as on over the employee's saved Off, which stays saved. Nothing else
  is granted: until its review (or the employee's skipped review), grants and dependencies admit
  it, and while another package shares its name, other bytes sit at the path or it is missing, the
  requirement is unmet, named in Skills with its reason and the process that saw it. Being
  required makes an extension run; it does not make it more trusted than its review and grants,
  and it does not isolate it. With `execution` it must also be trusted (Execution domain).
- `extensions.when_unmet` is required with any row. `hold` (the sample): queued work stays queued
  while the server's own copy does not run a requirement, and each process that starts work (a
  pool worker or the server) checks the extensions it has loaded itself; a turn it cannot serve
  ends before any model call with a typed notice (`deployment_requirement_unmet`). Stop, Panic,
  Settings and Skills stay available. `warn`: work goes on and the gap is shown. Either way each
  process logs a change of its gap once (`deployment_requirement_unmet`, `deployment_requirement_met`
  in `logs/events.jsonl`); nothing polls, retries or repairs.
- `settings` fixes custom settings (`A_KEY`, never a built-in or `OUROBOROS_` one) for that
  extension. They are read like any custom setting: listed in its manifest's `env_from_settings`
  (else the requirement is unmet) and granted by the employee for these bytes, so a fixed value is
  no permission to read it. Through that grant they reach the extension in process, in an
  out-of-process child (which receives the policy's path and pin after its environment is
  scrubbed) and in its companions' environment (which their manifest `env` cannot override), never
  the agent's shell or other environments. Settings shows them as set with no part of the value;
  no reveal or host error exposes one: load and runtime exceptions are told by type without their
  words/tracebacks. Successful plugin content is unchanged; trusted code must not disclose values itself.
  A save that changes a fixed value is refused.

Supported doors refuse, before any unload or write, to turn a required package off, delete or
uninstall it, update it (Replace or Merge), roll a kept update back, replace it or add a second
package of its name, hand it to a repair task, or edit its files with the agent's file tools or
the Files page (`managed_by_deployment`). A kept update stays pending and its review still decides
whether it runs. A missing required name may still be installed; it meets the
requirement only with the pinned bytes at the pinned path. Ordinary unload, shutdown, Stop and
Panic unload it as before. Its administrator changes the package and `content_hash`, then
restarts.

The policy is a view over settings, never their content. `settings.json` keeps the
employee's own choices. Readers apply the policy last, over task snapshots, rooms, live route
views and the environment. A save that carries a managed value back as Settings showed it
keeps the employee's saved choice; any other change to a managed choice is refused before
anything is written (HTTP 403, typed `managed_by_deployment`), on every Settings, owner, wizard
and launcher path, and the rest of Settings stays editable. `GET /api/settings` names what
the file fixes in `_meta.policy_state.deployment`, without secrets; Settings, the setup wizard
and model-wait cards offer only what it allows and show what it fixes (a level, the checker, its
connections or the fields its built-in routes read, a local checker's server, closed Safety exits,
the update channel) as set by the organization.

The serving program enforces the file, so within a managed deployment only its administrator
replaces it and its packages, outside Ouroboros (a new image, package or checkout). In-app Update
and its channel, Restore, recovery points, body adoption (evolution's included), agent edits that
resolve to the serving checkout and package installs into the serving interpreter (the local
runtime, the browser tools' Playwright package) refuse with `managed_by_deployment` before any
effect. Restart starts the same program: no checkout, reset, clean, dependency sync (server or
launcher) or recovery-point fallback runs, and a failed import is reported, not repaired. Update
and switch records from before wait untouched (one left mid-merge defers Restart); an adoption
armed before closes at the next start with the tree untouched, its candidate kept and the owner
told, while an interrupted one stops the start (exit 3) until the administrator decides the
tree. A first boot still prepares the seed's runtime venv. Body candidates, other workspaces,
installed or downloaded browsers, Stop and Panic stay.

A server reads the file once at its start and pins the digest of those bytes
(`OUROBOROS_DEPLOYMENT_POLICY_SHA256`) for every process it starts. A changed file applies at
the next start; until then a process started under the old pin refuses. Nothing polls the
file or copies it.

### Execution domain

`execution` names where employees' work runs, apart from the service that enforces the policy. The
reference shape is the service on a trusted Linux host and the work in containers of its own:

```json
"execution": {
  "executor": {"type": "docker_exec", "container_name": "work", "network": "host",
               "path_mappings": [{"host_path": "/srv/work", "backend_path": "/srv/work"}]},
  "engine": {"descriptor": "/etc/ouroboros/work-engine.json"},
  "dedicated_containers": ["work", "work-engine"],
  "skill_state_root": "/srv/work/skill-state",
  "staging_root": "/srv/work/staging"
}
```

- `executor` is the only executor: commands, scripts, checks and services run in that container
  whatever a task names, and so do stdio MCP servers, skill scripts and dependency installs, their
  streams carried by the service's Docker CLI. One whose folder, or a path-valued variable it is
  given, lies outside the mappings is refused before anything starts, never run on the service. A
  mapping may not overlap the data root, the checkout, this file, the engine descriptor or the
  service's import path. Commands are read in the container: each interpreter must exist there at
  the path the service names, or on the container's PATH. The container needs `setsid` (usually util-linux);
  without group isolation no command or service is started. An unconfirmed group end keeps its custody record
  and is reported to the caller even after a successful script or MCP response. `network` is the executor's word:
  `none` is checked before every command the service starts there, a skill's folder preparation
  included, and refused while the container has a network; `host` only allows networking and checks
  nothing. Neither puts the work in the service's network namespace.
- `engine` is the Claudexor engine that owns native, tool-running sessions, reached through a
  descriptor the administrator writes on the service (`{"host", "port", "tokenPath"}`, loopback
  only, outside the data root and the checkout): forward the engine's control port there, and keep
  the work containers out of the service's network namespace. Raw model calls, the Safety checker's
  included, keep the service's own engine and accounts; native sessions use this engine's accounts.
  Each native run records this engine and its store before it is sent, and every later step of it
  (wait, cancel, answers, messages, continuation, retry, recovery, collection, cleanup) reaches that
  engine only: another engine's answer never settles it, and a policy naming another engine later
  leaves it refused, not moved. Its sessions read work paths as the service names them: map them at
  the same address, or the start is refused. Without `engine`, native sessions are refused.
- `skill_state_root` is a persistent work folder, mapped at the same address, where each skill's own
  processes keep their state (`<root>/<skill>`, made there as the container's user): its
  `OUROBOROS_SKILL_STATE_DIR`, Deno's write permission and Go's cache and builds. The service keeps
  a skill's review, grants and enablement in its own state folder and copies none of them there;
  nothing an earlier run left there moves. Without it, skill scripts are refused.
- `staging_root` is a work folder, mapped at the same address, where the service prepares what the
  work needs: private copies of work repositories and a native skill-payload run's copy (`copies`), a
  native run's read-only inputs (a session scratch its task owns and retires with it), session scratch
  and package download caches, private Git indexes (`tmp`), and each installed script skill's reviewed
  payload (`skills/<name>/<content hash>`, verified against the service's digests before every run and
  replaced if changed) with its dependencies built there. Without it, these are refused; nothing falls
  back to the service.
- `dedicated_containers` belong to this installation alone. Panic sends one `docker kill` to the
  identities this server measured at its start (at Panic itself, when it had none) and never waits
  for it: what Docker answers before the exit is logged, anything else stays unconfirmed. Nothing
  restarts them, so give them no restart policy that would.
- `extensions.trusted` (only with `execution`, up to 64) lists the extensions whose code the service
  runs, each by `name`, `path` and `content_hash` as a required one; every required package is one
  of them, and none lies in or holds a folder the work domain writes (a mapping,
  `skill_state_root`, `staging_root`).
  Any other extension, built-ins included (unlisted, elsewhere, other bytes), is never imported,
  cataloged, registered or launched: Skills says it doesn't run here (`deployment_untrusted`) and it
  cannot be turned on. A
  trusted one keeps the whole PluginAPI and every other gate (its switch, review, grants), and its
  staged copy must hash to the pin before anything is imported from it. Its files, its
  `.ouroboros_env` dependencies (never installed or rebound here) and the staged imports are its
  administrator's: doors that would change, remove or replace them refuse (`managed_by_deployment`),
  turning it off excepted. A new extension or version arrives with the next pins and start. Trust
  does not apply to instruction or script skills, MCP servers or native sessions: they follow the
  work domain's rules above.

A path belongs to the work by its spelling inside a mapping, whatever a link there resolves to now.
Every effect on it runs in the work container under the executor's custody (`work_domain`): file
writes, renames and removals, and every Git command the tools run over a work tree, including private
copies, capture, apply, `restore`, `revert` and network Git, so the repository's hooks, filters,
fsmonitor, credentials and ssh act as the work domain with each action's usual semantics. The service
reads work bytes it shows, searches or captures only through descriptor-held opens from the mapping's
root that follow no link (it needs read access to the work volume; a link or a permission on the way
makes it the container's read), and folders cross as tar streams it checks member by member. A
native run receives its inputs as bytes under `staging_root`; its outputs return as patches and
checked bytes; a skill-payload draft returns to the service's own apply, so installing it stays
behind the usual review and, for a trusted extension, the administrator. Copies and inputs need an
identical mapping. Only trusted extensions run on the service. Map no folder whose code the service
itself runs, such as an extension's skill or its dependencies, and mount the work volume
`nodev,nosuid`.
The file protects neither itself nor what enforces it: keep it, the checkout, the data root, the
engine descriptor and token, extension packages, dependencies and staged imports, and the
server's environment where employee and agent processes
cannot write.

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
  A deployment can require it and fix its `SENTRY_DSN` (Deployment Policy File);
  its widget shows what each process's transport observed, and a receiver that
  refuses or cannot be reached is shown there while work and the local logs go on.
  Failures that happen before the extension loads stay in the local logs only,
  as do the launcher's and the Claudexor daemon's.
