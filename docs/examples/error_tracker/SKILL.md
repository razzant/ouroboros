---
name: error_tracker
description: Example. Sends this process's ERROR log records to a Sentry-compatible tracker and shows what each process's transport observed.
version: 0.2.0
type: extension
entry: plugin.py
plugin_api: "2.0"
permissions: [net, read_settings, route, widget]
env_from_settings: [SENTRY_DSN]
---
# Error tracker (example)

Initialises a Sentry-compatible SDK that is installed in Ouroboros's own
environment, in every process that loads this extension: the server and each
pool worker. Frame locals, source context, default integrations and personal
data are off; every event passes through Ouroboros's secret redaction before it
leaves. The destination is the `SENTRY_DSN` setting it is granted, else the
process environment's variable. Its widget shows, per process, what its transport
observed. See README.md for installation, coverage and why each setting matters.
