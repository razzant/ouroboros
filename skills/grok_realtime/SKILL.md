---
name: grok_realtime
description: xAI Grok Voice provider for Live Conversation.
version: 0.1.0
type: extension
runtime: python3
entry: plugin.py
plugin_api:
  version: "2.1"
  capabilities: [register_realtime_provider]
permissions: [realtime_provider, net, read_settings]
env_from_settings: [XAI_API_KEY]
---

Registers `grok_realtime:grok` for ordinary host-selected context and tools.
Grok Voice 3.0 declares a conversation window of
80,000 tokens in Ouroboros's own estimate (characters / 4), based on operator
probes on 2026-10-21. The 231,828 o200k-token accepted input is separate evidence,
not a fit target. This package never selects or cuts memory. Host facts keep their roles;
`Respond` opens the greeting only after setup acknowledgements. Call instructions
follow host system text; language tags add a recognition sentence, not a session
field. Silent tool outputs do not request another response.
Selected history containing tool calls is carried as ordered, quoted JSON in
instructions, preserving arguments and results without reissuing those calls.

The provider reports modality usage when present, final response status, and
cumulative connected seconds. Idle samples and the closing tail reach the host
through the `observe("usage", ...)` callback; the host prices only growth within
one physical connection. Provider usage remains raw evidence beside those axes.
Server VAD decides pauses and interruptions. There is no image input; Show is
unavailable. The feature owns playback, UI and call renewal; the host owns tools,
history, shared spending and Stop. Credentials are read only on an admitted
connection. Installing or granting this provider does not enable Calls (beta).

Catalog checked 2026-10-21: thinking follows `reasoning.effort` (`none`/`high`);
`eve` is documented and `xai_ara` was observed as the server default on 2026-10-20.
The empty voice choice leaves the provider default in effect.
