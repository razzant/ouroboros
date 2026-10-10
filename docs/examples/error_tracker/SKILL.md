---
name: error_tracker
description: Example. Sends this process's ERROR log records to a Sentry-compatible tracker named by SENTRY_DSN.
version: 0.1.0
type: extension
entry: plugin.py
plugin_api: "2.0"
permissions: [net]
---
# Error tracker (example)

Initialises a Sentry-compatible SDK that is installed in Ouroboros's own
environment, in every process that loads this extension: the server and each
pool worker. Frame locals, source context, default integrations and personal
data are off; every event passes through Ouroboros's secret redaction before it
leaves. See README.md for installation, coverage and why each setting matters.
