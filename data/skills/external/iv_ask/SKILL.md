---
name: iv_ask
description: Ask the fixed Iv Hermes profile once and return only a bounded terminal result.
version: 0.1.0
type: extension
entry: plugin.py
runtime: python3
permissions: [tool, net, read_settings]
env_from_settings: [HERMES_API_KEY, HERMES_IV_ASK_ROOT_ID]
timeout_sec: 135
---

# iv_ask — Stage 1C

`iv_ask(text)` sends the original Lia text unchanged to the host-fixed Iv route.
The extension supplies trusted root provenance from owner-granted host settings;
the model cannot choose a profile, session, route, run ID, provenance, or URL.
Hermes starts exactly one Runs API run and exposes only a capped, redacted
terminal result (no reasoning, usage, events, or tool payloads).

The future live payload path is exactly
`/home/msheldyakov/ouroboros/data/skills/external/iv_ask/`.
Review the payload, grant `HERMES_API_KEY` and `HERMES_IV_ASK_ROOT_ID`, enable
`iv_ask`, then reload extensions. Hermes must already have the matching
`gateway.api_server.lia_iv_ask.root_id` and fixed `model` configured.
