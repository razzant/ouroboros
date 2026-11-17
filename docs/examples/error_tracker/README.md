# Send errors to a Sentry-compatible tracker

Verified on 2026-10-06 with sentry-sdk 2.71.0 against the Ouroboros source of the pull request that
added this example: an isolated server with two pool workers loaded the extension in every process, and
a local Sentry-protocol endpoint received the server's and both workers' errors with secrets masked and
no frame locals. The results are in that pull request. Verified again on 2026-11-08 with sentry-sdk
2.71.0 under a deployment policy that required this extension and fixed its DSN: the loader read the DSN
through the extension's grant, and a synthetic loopback receiver that answered HTTP 200, then HTTP 500,
then was gone left the widget's records at accepted, refused and unreachable while the logging calls
returned and the local log handlers kept every record (`tests/test_monitoring_examples.py`).

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
2. Give it a destination. Either save `SENTRY_DSN` in Settings → Accounts and keys and grant it to this extension
   in Skills (each process reads it at load), or set `SENTRY_DSN` in the server's environment, which
   pool workers inherit and the SDK reads itself. Without either it sends nothing.
3. Copy this directory to `data/skills/external/error_tracker/`, review it (or skip the review as the
   owner), enable it, and restart, so that workers already running load it too.

## Requiring it

A deployment policy file can require this extension and fix its destination (`docs/DEPLOYMENT.md`,
Deployment Policy File): name it, its directory and the `content_hash` Skills reports for it, and put
`SENTRY_DSN` in its `settings`. It then reads as on and its card has no Off, removal or update; it still
runs only after its review and the employee's grant of `SENTRY_DSN`, and while it does not run in a
process (the SDK missing, a malformed DSN, no grant) Skills says why and the file's `when_unmet` decides
whether ordinary work waits there (`hold`) or goes on (`warn`). Requiring it does not make the extension
more trusted than its review and grants. That requirement status is Ouroboros's own; what the tracker
delivers is this extension's widget, and neither stands in for the other.

## What its widget shows

Each process that loaded the extension writes what its own transport observed to
`state/skills/error_tracker/delivery/<role>-<pid>.json`; the Error tracker widget reads at most 16 most recently
updated records, says how many it leaves out, and states nothing about those. Each row names the process (server
or worker, its pid, and running, ended or liveness unknown) and where its destination came from
(Settings, Environment, Not configured). A process is ended only when it is provably gone or its pid now
belongs to a later process; one whose start time cannot be read is of unknown liveness and still counts
when its last report was not accepted. No DSN or event text is stored there. The directory metadata walk still
grows with retained process history; JSON reads and sort memory are bounded, and no records are deleted.

- Queued: an error report handed to the SDK's send queue. Queued is not sent.
- Accepted: the receiver answered HTTP 2xx. The receiver took the request; that says nothing about
  whether it stored or processed the event.
- Refused: another HTTP status, kept with the status.
- Unreachable: no HTTP answer (refused connection, timeout, DNS).

The table counts these four. Two more outcomes are counted in each record and appear as a process's
last outcome when they are: rate limited (HTTP 429) and dropped (lost before sending because the
queue was full).

A receiver that refuses or cannot be reached is shown here and nothing is resent; work goes on and the
local logs keep every record. An event still queued when its process ends is lost. The counts start
when the extension loads in that process; a process that ended keeps its last record, marked ended.
The transport hooks are sentry-sdk 2.71.0's own HTTP transport methods (`capture_envelope`,
`_handle_response`, `_handle_request_error`, `on_dropped_event`), private to that version: another SDK
version that lacks one reports less, and the status route names the SDK version each process loaded.

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
- `transport` is a subclass of the SDK's HTTP transport, passed as a class the SDK instantiates: it
  records the outcomes above and changes nothing it sends.

## What it does not see

- Failures before the extension loads.
- The desktop launcher.
- Out-of-process extension children.
- The Claudexor daemon and the coding tools it runs.
- An event still queued when a process is killed.

Those stay in the local logs (`logs/server.log`, the workers' stderr, `logs/launcher.log`). The record
passport, [`ouroboros/contracts/record_contract.py`](../../../ouroboros/contracts/record_contract.py),
lists what the JSONL records guarantee.
