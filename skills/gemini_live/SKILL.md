---
name: gemini_live
description: Gemini native Live audio and image provider for Live Conversation.
version: 0.1.14
type: extension
runtime: python3
entry: plugin.py
plugin_api:
  version: "2.1"
  capabilities: [register_realtime_provider]
permissions: [realtime_provider, net, read_settings]
env_from_settings: [GEMINI_API_KEY]
---

Registers `gemini_live:gemini`. It reuses the preserved Gemini Live transport
mechanisms: native audio, function responses, caption accumulation and explicit
fresh-context renewal. The full ordinary selected history is an ordered role
archive in Gemini's system layout; this is provider packaging, not a separate
memory or summary. An unexplained drop in the speech Gemini retains (its usage
shows earlier audio gone without a compression that explains it), or a
tool-catalog change, asks the host for a fresh refit after pending work
finishes. The latest unexplained drop is the learned span; there is no fixed
span. Context-window compression is on: the catalog window triggers it and the
window minus the working reserve is its target. The host also owns usage-driven
total-window renewals; both use the same quiet boundary. Idle time alone does
not renew. GoAway asks for server-side resumption.
Session resumption is enabled; handle updates reach the feature, and transport
loss is marked resumable only while a valid handle exists; otherwise the shared
feature can continue through a fresh fitted session. Provider refusals remain terminal. A replacement using
that handle restores in-process caption, tool, held-context and usage state.
Completed response ids, timestamped transcripts and function-call ids prevent
identified replays from being counted twice; identical words without native
identity are never guessed to be a replay. Reconnecting state is consumed once,
expires with Google's two-hour handle lifetime and is never written to disk.
The feature must retain playback acknowledgements and defer partial-caption and
response settlement across a resumed connection; Stop still finalizes them.
Native VAD and input transcription open input work; sent PCM (including silence)
does not. Unfinished captions and unanswered tools keep renewal pending.
Late output captions retain their audio item's identity; generation completion
alone cannot release that caption barrier. Extended thinking finalizes received
text at turnComplete/IDLE; later transcription tails emit a full replacement
final for the same item, including across resumption. A local schema change wakes the
receiver without cancelling its socket read. Expiry with pending work ends
explicitly; Stop retains any known unfinished text as a partial transcript.
Call instructions follow the host system text. Language hints, voice, pauses,
interruptions and optional Google search follow LiveOptions. Extended thinking
uses the selected level (low by default); the ordinary model has interleaved
reasoning without a level control. Respond opens a host-labelled greeting fact.
Silent tool results use native SILENT scheduling on the ordinary model; the
extended-thinking model does not support scheduling, so the feature must finish
hangup locally. Tool-schema changes coalesce into one pending fresh renewal.
The catalog offers both models and 30 prebuilt voices. Its 92k o200k window is
measured for the ordinary model and a labelled hypothesis for extended thinking;
the documented 131,072 provider-token input limit is a separate quantity.
Every completed response reports modality usage, cached input and thinking units
where Gemini supplies them, with raw evidence and status. Sparse modality lists
prove omitted zeroes when their counts cover the reported total, or for an input
modality never submitted to this session. Selected images and retained media
across resumed connections count as submitted; an unknown resume has no such
absence evidence. Missing lists, unaccounted modalities and missing cache or
thinking counts stay unknown. Reported thinking tokens are billed in addition
to response modalities; an explicitly empty cache list proves zero cached tokens.
The provider accepts inline images and sends 16 kHz mono PCM input / 24 kHz output.
Installing or granting it does not enable the Live Conversation beta.

Wire reference: https://ai.google.dev/api/live (updated 2026-10-09 UTC), model
pages and capabilities guide (read 2026-10-21). Live-provider probes remain
necessary to verify the deployed dialect and the labelled retention hypotheses.

This provider declares silent context updates: fitted catch-up history with
`respond=False` does not open an answer, allowing the host to prepare a successor.

Host-only bookkeeping is absent from the structured wire projection; content strings stay exact. Delivered setup and update rows become exposure evidence only at a usable response, before queued updates are released. Host fitting owns the shared remedies and exact physical-cut checkpoints; the provider does not choose a memory horizon.
