# Host the interface apart from the core, behind one TLS origin

Verified on 2026-10-13 with Caddy 2.11.7 (macOS arm64), Chromium and WebKit (Playwright 1.62) by
[`tests/test_separate_ui_browser.py`](../../../tests/test_separate_ui_browser.py). The test runs these two
site blocks with a real Ouroboros core, a copy of the builtin `web/` tree on the interface host and
a certificate made for the run. Test-only global Caddy options and the forwarding change are described
under "What the test changes"; this is not an unchanged two-process administration test.

The core keeps its state, tasks, sign-in, socket, setup wizard and extension resources. The interface is
only files: the builtin `web/` tree or your own build laid out like it. A browser sees one origin:

```text
browser ──TLS──> front proxy (Caddyfile)
                   ├─ /api, /auth, /onboarding, /ws ──> core (server.py, OUROBOROS_UI_DIR=none)
                   └─ everything else ────────────────> interface host (interface.Caddyfile, static files)
```

Each piece starts and stops on its own. The page does not reload when the core restarts; it waits,
reconnects and reads its state and history again. While the interface host is down, an open page keeps
working, and only loading `/` fails.

## The core

- `OUROBOROS_UI_DIR=none`, as an environment variable or a saved setting, so the core serves no main page.
  See "Choosing what the core serves at `/`" in [`docs/CLIENT_CONTRACT.md`](../../CLIENT_CONTRACT.md).
- `OUROBOROS_NETWORK_PASSWORD`. The core asks for it only when the visitor is not loopback.
- Keep the core on its default `127.0.0.1` bind when the proxy runs on the same machine. A proxy on another
  machine must be named in uvicorn's `FORWARDED_ALLOW_IPS`. Otherwise the core ignores its
  `X-Forwarded-For` and `X-Forwarded-Proto`.
- `OUROBOROS_FILE_BROWSER_DEFAULT`, a directory, if visitors use Files. A visitor who is not loopback gets
  400 from `/api/files/*` without it.

## The interface host

Copy the `web/` tree of the same Ouroboros version, or your own build. Then run:

```bash
OUROBOROS_INTERFACE_ROOT=/srv/ouroboros-web caddy run --config interface.Caddyfile
```

It answers `/` with `index.html`, `/static/*` with the files and everything else with 404. It binds
`127.0.0.1:8080` by default (`OUROBOROS_INTERFACE_BIND`, `OUROBOROS_INTERFACE_PORT`). It never needs core
code, settings or data. The core's setup wizard (`/onboarding/static/*`) and the author kit always come from
the core, whatever the interface host serves.

With `OUROBOROS_UI_DIR=none`, `/onboarding` and `/auth/login` keep the builtin pages.
Your interface host may provide its own setup/sign-in at interface addresses and use
`/api/onboarding/bootstrap`, `/api/onboarding/complete` and JSON `POST /auth/login`.
The core does not read optional pages from that other host. To have the core itself
serve a complete client, select its local build directory instead; optional setup/login
files and recovery are described in [CLIENT_CONTRACT §2–4](../../CLIENT_CONTRACT.md)
and the [minimal client example](../client_views/README.md).

## The front proxy

```bash
OUROBOROS_SITE=https://ouroboros.example.lan \
OUROBOROS_TLS_CERT=/etc/ouroboros/cert.pem OUROBOROS_TLS_KEY=/etc/ouroboros/key.pem \
OUROBOROS_CORE_UPSTREAM=127.0.0.1:8765 OUROBOROS_INTERFACE_UPSTREAM=127.0.0.1:8080 \
caddy run --config Caddyfile
```

- For a public name that Caddy can certify itself, delete the `tls` line.
- The `@core` matcher must list exactly the core's `core_paths` from `GET /api/contract`, each with its
  `/*` form. The test compares them.
- `header_up X-Forwarded-For {client_ip}` overwrites what the visitor sent with the peer Caddy saw.
  Do not trust a visitor-supplied forwarding claim; align the core's trusted proxy list with this topology.
- Caddy forwards `X-Forwarded-Proto: https`, so the core marks its session cookie `Secure`. WebSocket
  upgrades on `/ws` pass through without extra settings.
- Access logs go to Caddy's log, one line per request. `log_append` names the upstream that answered, and
  Caddy writes `Cookie` and `Authorization` values as `REDACTED`.

A visitor on the proxy's own machine reaches the core as loopback, and the core does not ask loopback for
the password. That matches the core's own rule for local use. Do not run a browser for other people on the
proxy machine.

## What the test changes, and what it proves

The test browser runs on loopback, so with the published `Caddyfile` the core would never ask it for the
password. The test therefore replaces `{client_ip}` on the `X-Forwarded-For` line with `198.51.100.23`, a
documentation address. A separate test keeps the published site block unchanged and checks that a
visitor's own `X-Forwarded-For` is overwritten. Both also prepend global options: administration and
config persistence off, automatic HTTP redirects off, loopback binding, run-local storage, and HTTP/1.1
and HTTP/2 only. Thus shared administration ports, normal persistence, redirect listeners, non-loopback
binding and HTTP/3 are not qualified by these tests. The test trusts its certificate
only inside its own browser: by key pin in Chromium, and through the WebKit context ignoring certificate
errors. In both engines it checks the certificate the page was served with.

Proven there, in Chromium and WebKit unless noted:
- sign-in from an empty cookie jar, a wrong password, the return to the interface with HTTP and WebSocket;
- projects, a chat message and the task the core started for it, history;
- a proxy restart, a core stop and start, an interface host stop and start;
- uploads and downloads, identical by SHA-256;
- module and iframe widgets and the author kit;
- the following in Chromium only: Stop, questions answered live and late, Settings and their restart flag,
  update status and preflight, Restart and Panic, and a fresh install set up through the core's wizard.

Not covered: a proxy on another machine, ACME, HTTP/3, Firefox, mobile shells. A module widget's direct
(non-bridge) loads are 401 in Chromium, but WebKit sent the session cookie with them in this test.

## Try it from a source checkout

`python -m tests.separate_ui_support` starts the same three processes on loopback with a synthetic
password and prints the address, the password and Chromium's certificate flag. Ctrl-C stops all three. The
test module's docstring has the exact command.
