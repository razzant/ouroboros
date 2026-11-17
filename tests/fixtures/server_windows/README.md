# Server window metadata fixtures

| File | Server, version | How obtained |
|---|---|---|
| `llama-b12421-v1-models.json` | llama.cpp b12421-75fd7f9ec, `-c 80000 -np 1` | Real `GET /v1/models` on the project stand (stand/evidence/http/a-models), 2026-11-15 |
| `llama-b12421-props.json` | same | Real `GET /props` (a-props) |
| `llama-b12421-slots.json` | same | Real `GET /slots` (a-slots) |
| `llama-b12421-np1-80k-props.json` | llama.cpp b12421, `-c 80000 -np 1`, default cache | Real `GET /props` (stand/evidence2/cache-np1-80k); n_ctx 80128 = 80000 padded to 256 |
| `llama-b12421-np2-80k-props.json` | llama.cpp b12421, `-c 80000 -np 2` | Real `GET /props` (cache-np2-80k): the slot window is 40192, half the buffer |
| `vllm-0.33-v1-models.json` | vLLM 0.33 | Reconstructed from `entrypoints/openai/models/serving.py` (`max_model_len = self.model_config.max_model_len`), reports/W2b §2 |
| `lmstudio-0.4-api-v0-models.json` | LM Studio 0.4.x | Reconstructed from the REST v0 docs (`loaded_context_length`, `state`), reports/W2a §4 |
| `lmstudio-0.4-v1-models.json` | LM Studio 0.4.x | Reconstructed: the OpenAI-compatible listing has no window field |
| `ollama-0.40-api-ps.json`, `ollama-0.40-api-ps-empty.json` | Ollama v0.40.16 | Reconstructed from `api/types.go` (`ContextLength int json:"context_length"`), reports/W2a §3; empty before a model is loaded |
| `ollama-0.40-v1-models.json` | Ollama v0.40.16 | Reconstructed: the OpenAI-compatible listing has no window field |
