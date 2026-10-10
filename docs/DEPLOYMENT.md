# DEPLOYMENT.md — Deployment Notes

## Container Restart Policy and Panic

The Docker image starts `server.py` directly. Set
`OUROBOROS_LAUNCH_INTENT=automatic` and use `restart: on-failure` so an
automatic restart preserves Panic. The server uses the launcher's existing
Panic check before runtime startup: it leaves the stop marker intact and
exits 0. Panic itself still exits 99; Docker may restart it once, then the
automatic entry exits cleanly and `on-failure` leaves it stopped.

For an image built as `ouroboros-web`, a Compose service can use:

```yaml
services:
  ouroboros:
    image: ouroboros-web
    ports:
      - "127.0.0.1:8765:8765"
    environment:
      OUROBOROS_DATA_DIR: /data
      OUROBOROS_LAUNCH_INTENT: automatic
    volumes:
      - ouroboros-data:/data
    restart: on-failure
volumes:
  ouroboros-data:
```

An absent intent or `OUROBOROS_LAUNCH_INTENT=owner` keeps explicit owner startup
available. To resume the stopped example with the same data volume:

```bash
docker compose run --rm --service-ports -e OUROBOROS_LAUNCH_INTENT=owner ouroboros
```

This runs a foreground owner session; `--rm` disables its restart policy and
removes that container on exit, while the named data volume survives. An
ordinary `docker start` retains the configured automatic intent and therefore
does not resume Panic. Do not delete the stop marker to resume.

`always` and `unless-stopped` can keep restarting the process after its clean
exit; the marker stays intact, but these policies do not provide the supported
stopped-container behavior. Without automatic intent, restart policies can
still consume the marker and resume the server. See Docker's
[restart policy documentation](https://docs.docker.com/engine/containers/start-containers-automatically/).

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
- The durable records stay the JSONL ledgers under `logs/` and the task drives
  (`docs/PERSISTENCE.md`); a file-tailing collector (OpenTelemetry Collector,
  Vector, Fluent Bit) can ship them without any change to Ouroboros. The subset
  an observer may rely on is the record passport,
  `ouroboros/contracts/record_contract.py`: the anchor rows and their
  guaranteed fields, correlation ids, which fields carry content, rotation, the
  tools-row replica, the child task drives that are deleted after
  `OUROBOROS_GC_RETENTION_DAYS`, and where money must be read (the accounting
  views, never a sum of `llm_usage`). `docs/examples/log_collector/` is a
  verified Vector setup that ships their metadata only.
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
