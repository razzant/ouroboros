# Local server reasoning fixtures

| File | Server, model | How obtained |
|---|---|---|
| `llama-b12421-qwen35-9b-think-tool-call.json` | llama.cpp b12421-75fd7f9ec, `--reasoning-format none`, Qwen3.5-9B Q8_0, thinking on | Real streamed `POST /v1/chat/completions` reply on the project stand (pair 00472, series b6-raw-file, 2026-11-16), its SSE deltas joined into one message; `source_sha256` is the raw stream's. Only the stand's absolute folder was replaced by `/stand` |
| `llama-b12421-qwen35-9b-think-text-answer.json` | the same server and model | Real streamed reply with a leading `<think>` and a text answer, no tool calls (pair 00045, series s2-1, 2026-11-17), deltas joined the same way; it holds no local paths |
