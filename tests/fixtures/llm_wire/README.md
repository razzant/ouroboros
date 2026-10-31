# Recorded provider wire

Real SSE streams and non-stream JSON bodies, recorded from live provider routes on
2026-09-13 with synthetic prompts (an etymology question that must call one `lookup`
tool once or twice), and one structure-only export of a real task's stream (below). They are the offline evidence behind `tests/test_llm_wire_corpus.py`:
the stream assembler (`ouroboros/llm_stream.py`) must consume every `.sse` here without an
exception and produce the same normalized shape as the `.json` sibling of the same request.

Layout: `<route>/<model>/<case>_stream.sse` and, where the same request was also sent
non-streaming, `<case>_nonstream.json`. `route` is the transport family (`openrouter`,
`openai` Chat Completions, `anthropic` native Messages), never a vendor-specific rule.

How they were recorded: a plain `curl -N`-shaped POST (`urllib.request`, no SDK) of the
Chat Completions / Messages request with `"stream": true` (plus
`"stream_options": {"include_usage": true}` on the OpenAI-compatible routes), the raw
response bytes written to disk unchanged. The sprint's throwaway `capture_wire.py` and
`replay_corpus.py` did exactly that; the production runtime also retains every stream's
raw wire in the private CAS `physical_stream` manifest (`_StreamAssembly.retain`), which is
the source for future exports.

Redaction rule (applied before commit): every string value under the keys `data` and
`signature` (opaque encrypted reasoning / thought signatures) keeps its first 24
characters followed by `…`. Ids (`call_…`, `rs_…`, `gen-…`, `toolu_…`), types, indices,
text, tool arguments and usage are intact. SSE framing is byte-exact (`: OPENROUTER
PROCESSING` comment lines, blank lines, `[DONE]`, `event:` lines, the provider's own JSON
spacing on frames the redaction did not touch); a redacted `data:` line is re-serialized
compactly. Non-stream bodies keep their leading keep-alive whitespace.

One exception: `claude-opus-5.5/twoblock_stream.sse` comes from a real task's retained
`physical_stream` (2026-10-20), because no synthetic prompt drew two thinking blocks. There
every text and tool-argument value is filler of the same length (keys kept) and ids are
fixture ids; frames, indices, signature prefixes and usage are as recorded.

Fixture pairs are separate live requests, so parity is structural (message keys, tool
names, argument key sets, `reasoning_details` type sequence and per-record key sets,
finish reason, usage keys), never exact text.

Cases — the same synthetic prompt and `lookup` tool everywhere except `twoblock_stream`;
what differs is the request knob (or simply the reply the model chose to give on that run):

| case | route / model | request knob | what the reply exercises |
|---|---|---|---|
| `tool_stream` (+ `tool_nonstream`) | openrouter / gemini-3.8-flash | `reasoning: {"enabled": true}` | one call, a single `reasoning.encrypted` record |
| `tool_stream_longreasoning` | openrouter / gemini-3.8-flash | `reasoning: {"effort": "low"}` | one call; visible `reasoning.text` then `reasoning.encrypted` — the #856 minimal reproducer (type transition inside one `index`) |
| `multicall_stream` (+ `multicall_nonstream`) | openrouter / gemini-3.8-flash | `reasoning: {"effort": "high"}` | two calls in one turn |
| `secondturn_stream` | openrouter / gemini-3.8-flash | `reasoning: {"effort": "high"}`, second turn (assistant reply + `lookup` result replayed) | continuation with replayed reasoning |
| `tool_stream` | openrouter / grok-4.6 | `reasoning: {"effort": "low"}` | `reasoning.summary` then `reasoning.encrypted` (`rs_…`) |
| `tool_stream`, `multi_stream` | openrouter / gpt-5.6-sol | `reasoning: {"effort": "low"}` / `{"effort": "medium"}` | one / two calls, one `reasoning.encrypted` |
| `twoblock_stream` | openrouter / claude-opus-5.5 | real task, `reasoning: {"effort": "max"}` | two signed id-less `reasoning.text` blocks (`index` 0, 1), then two calls |
| `tool_stream` (+ `tool_nonstream`) | openrouter / claude-sonnet-5 | `reasoning: {"effort": "high"}` | two calls, `reasoning.text` with a signature |
| `refusal_stream`, `refusal_stream_v2` | openrouter / claude-fable-5 | `reasoning: {"effort": "high"}` | a refusal: `finish_reason: content_filter`, no reasoning |
| `custom_stream` (+ `custom_nonstream`) | openai / gpt-5.6-terra | `tools[0].type = "custom"` with the runtime's `_CUSTOM_FORMAT` | a custom-tool call |
| `function_none_stream` | openai / gpt-5.6-terra | `reasoning_effort: "none"` | two function calls, no reasoning |
| `native_stream` (+ `native_nonstream`) | anthropic / claude-sonnet-5 (Messages) | `thinking: {"type": "adaptive"}` | thinking + text + two `tool_use` blocks |
