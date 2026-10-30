---
name: live_conversation
description: Talk with Ouroboros using this room's ordinary memory, tools and controls.
version: 0.1.23
type: extension
runtime: python3
entry: plugin.py
plugin_api:
  version: "2.2"
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

With providers declaring silent context updates, fresh renewals prepare the next
fitted connection while this one keeps listening; older providers renew sequentially.
At a quiet moment the call switches with new canonical facts intact and no second
greeting. A transport loss without a native handle continues from a fresh fit;
provider refusals still end typed, and repeated unanswered drops remain bounded.

The waiting successor keeps reading and metering into the shared limit; its events
wait for adoption. Overlap has no separate time bound and adds connected seconds
at the captured route tariff. Current options and account scope are checked at
the switch. Transport failure or a stale fit falls back; provider refusals end
without another attempt. Committed room mail keeps its answer opportunity across
a handoff, and an interrupted answer proves the connection just as a completed one does.

Fresh candidates use the shared transcript remedies before the final physical window cut: oldest whole rows of any kind move to exact checkpoint addresses; core, assignment, notes, newest owner turn and current tail stay. A failed speculative fit keeps the serving connection only while pending deliveries fit its measured free room; otherwise the call ends with context_capacity, keeping receipts and completed effects. A change from the failed snapshot gets a fresh preparation before old-socket delivery is considered. A complete ordinary result that exceeds serving room waits in that fresh fit; if it fails, the ordinary measured range view is the fallback, without repeating the effect. In the fresh call projection, owner Nano batches more than four images to two; Max/Low keep the ten-to-five batch. Canonical call images stay intact during preparation; adoption and call end apply the same floor only to images with exact saved paths. Pixels already in the fitted prefix are not sent twice. Earlier same-source frames are captioned by the fresh engine projection; canonical originals remain readable.
