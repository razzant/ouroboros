---
name: live_conversation
description: Talk with Ouroboros using this room's ordinary memory, tools and controls.
version: 0.1.0
type: extension
runtime: python3
entry: plugin.py
plugin_api:
  version: "2.1"
  capabilities: [register_live_conversation, register_route, register_settings_section]
permissions: [live_conversation, route, widget, page_module]
contributes:
  - id: call
    provides: conversation
    runtime: client
    label: Live Conversation
    destination: host
    render: {kind: host, entry: ui/index.js}
---

Calls are a beta, off by default. The call skills are on by default. To talk,
turn on Calls (beta), choose an engine and add its API key in Settings → Secrets,
under the keys the skills request.
Gemini, Grok and OpenAI supply the engine, model and voice choices from their
loaded catalogs. The owner's skill auto-grant setting still applies; with it off,
requested permissions await approval in Skills.
Having a key does not turn on calls; an unavailable selection is never replaced.
The selected catalog supplies supported controls, privacy and memory-window
facts. Call instructions and engine choices apply to the next call. Microphone,
speaker, camera and echo/noise/gain choices stay on this client and apply live.

Talk occupies Send while the composer is empty. Text or a file restores Send;
empty Enter never starts a call. Dictation is hidden during the call. Keyboard
returns to the same draft. Hidden rooms and documents keep the call where the
platform allows. End, ordinary Stop, Panic, feature/provider unload, pagehide,
device loss and confirmed Calls-off end it. Failed config reads keep it.

The model reasons over the ordinary selected context and calls ordinary tools.
The host owns memory, canonical roles and history, authority, money and device
leases. Each engine owns its physical wire layout, captions, usage and renewal.
The ordinary shared spending limit is the only money limit; there is no call
time cap or per-call reserve. The driver uses only LiveHost, greets once through
Respond, and renews at a quiet boundary. Its call-local tools remember BCP-47
recognition languages per owner/Presence binding and end after goodbye playback.
The engine-neutral Live calls instructions belong to this feature and precede
the owner's call instructions on every connection.
