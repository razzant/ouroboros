---
name: openai_realtime
description: OpenAI native Realtime audio, images and tools over the host's ordinary selected context.
version: 0.1.9
type: extension
runtime: python3
entry: plugin.py
plugin_api:
  version: "2.1"
  capabilities: [register_realtime_provider]
permissions: [realtime_provider, net, read_settings]
env_from_settings: [OPENAI_API_KEY]
---

Registers `openai_realtime:openai` for `gpt-realtime-2.1`. The host selects
memory, roles, tools and output allowance. The provider owns wire layout,
acknowledgements, captions, usage axes and its catalog. The route's window is the
measured lower bound: 133,554 input tokens accepted and answered on 2026-10-20
(documented context 128,000); spoken retention is still unmeasured. Automatic
truncation is disabled.
Text fields preserve every character and role across the observed 256000-character
item limit. Images remain native input. Unsupported content fails explicitly.

Call instructions follow host system text. The first language tag supplies the
ISO-639-1 transcription hint; additional tags stay in instructions. Pause maps
to server-VAD silence duration (300/500/1000 ms); interruptions follow the owner's
option. Responses remain host-driven, after setup and context/tool ACKs. `Respond`
opens the greeting; silent tool outputs request no answer.

The native per-response limit encodes allowances up to 4096 directly; larger
allowances use `"inf"` (the model maximum). An unknown model maximum stays unknown.
Usage reports text/audio/image/cache input and text/audio output independently,
with raw usage and completion status, including interrupted or failed responses.
Reasoning stays in raw usage and is already included in text output, never charged
again as separate thinking tokens. Video has zero usage; frames are billed as images.
There is no diagnostic probe or per-connection reservation in this package.

Renewal opens a fresh host-refitted session; there is no resumption. Native
transcription, response and tool/ACK work must finish first (`renewal_ready`);
the feature also waits for playback and durable caption writes. The host owns
history, tools, shared spending and Stop. Credentials are read only while opening
an admitted connection. Granting a key does not enable Calls (beta).

Catalog checked 2026-10-21: thinking follows `reasoning.effort` (minimal through
xhigh); the ten built-in voices remain available.

This provider declares silent context updates: fitted catch-up history with
`respond=False` does not open an answer, allowing the host to prepare a successor.

Host-only bookkeeping is absent from the structured wire projection; content strings stay exact. Delivered setup and update rows become exposure evidence only at a usable response, before queued updates are released. Host fitting owns the shared remedies and exact physical-cut checkpoints; the provider does not choose a memory horizon.
