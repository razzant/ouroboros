# Send errors to a Sentry-compatible tracker

Verified on 2026-10-06 with sentry-sdk 2.71.0 against the Ouroboros source of the pull request that
added this example: an isolated server with two pool workers loaded the extension in every process, and
a local Sentry-protocol endpoint received the server's and both workers' errors with secrets masked and
no frame locals. The results are in that pull request.

Ouroboros sends no telemetry. This example is for a deployment that runs its own error tracker
(self-hosted Sentry, GlitchTip, or any service that accepts the Sentry protocol).

## What it sends

For each ERROR or CRITICAL record of a process that loaded it (the SDK skips a repeat of an exception it
has already sent): the message, the exception type and text, the code locations of the stack (file paths
and function names), the logger, the level, the process role and the Ouroboros version. Secret shapes are
masked, but task text, file paths and other user content inside a message or an exception are not, and a
secret Ouroboros does not recognize passes too. Enabling this example sends that diagnostic content to
your tracker; treat the tracker as holding operational data.

## How it reaches the errors

Every process Ouroboros configures logs an unexpected failure as a stdlib `logging` record with its
traceback: a task exception, a worker crash, an unhandled gateway request (HTTP 500), an uncaught thread
exception. The SDK's logging integration turns each ERROR or CRITICAL record into an event. The extension
starts the SDK in the server and in every pool worker, because it has no declared dependencies and no
binary files, so Ouroboros loads it in process.

## Install

1. Install `sentry-sdk` into Ouroboros's own environment: in Docker, one layer of your image
   (`pip install sentry-sdk`). Do not declare it as a skill dependency: a declared dependency moves the
   extension into a separate process, where it cannot see these records. The packaged desktop app has
   no supported way to add a package to its interpreter.
2. Set `SENTRY_DSN` in the server's environment; pool workers inherit it. Without it the SDK sends
   nothing.
3. Copy this directory to `data/skills/external/error_tracker/`, review it (or attest it as the
   owner), enable it, and restart, so that workers already running load it too.

## Why each setting

With the SDK's defaults, a pilot sent a fake API key verbatim twice, in the exception text and in a
frame's local variables. It also sent the machine's hostname, the command line and the list of
installed modules, and it enabled its integrations for the HTTP and model clients (httpx, openai),
which add trace headers to outbound requests. The example turns all of that off:

- `include_local_variables=False` and `include_source_context=False`: frame locals and source lines
  hold prompts, keys and file contents.
- `default_integrations=False` with only `LoggingIntegration`, `DedupeIntegration` and
  `AtexitIntegration`: no automatic instrumentation of HTTP or model clients, and
  `trace_propagation_targets=[]` keeps trace headers off every outbound request.
- `send_default_pii=False`, `max_request_body_size="never"`, and `server_name` set to the process role
  instead of the hostname.
- `before_send` keeps only the fields in `_EVENT_FIELDS` (the event's identity and time, level, logger,
  platform, SDK, process role, release, environment, tags, fingerprint, message and exception), dropping
  `extra` (the logging integration copies every key a caller passed as `extra=`) and everything else,
  and passes what it keeps through `ouroboros.observability.redact_projection`, Ouroboros's own secret
  redaction.

## What it does not see

- Failures before the extension loads.
- The desktop launcher.
- Out-of-process extension children.
- The Claudexor daemon and the coding tools it runs.
- An event still queued when a process is killed.

Those stay in the local logs (`logs/server.log`, the workers' stderr, `logs/launcher.log`). The record
passport, [`ouroboros/contracts/record_contract.py`](../../../ouroboros/contracts/record_contract.py),
lists what the JSONL records guarantee.
