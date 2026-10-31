# Offline engine wire

Captured through production `ClaudexorGateway` and `LLMClient` by
`test_claudexor_capability_engine.py`. Engine HTTP bodies are real output.
Identities, JWTs and injected provider responses are synthetic; no real provider
or operator account is used. Uploads, keys and provider requests are recorded.

Candidate transcripts carry tree `94eaa9ef8cb232b3d3a4b31628c9e1a9c1b5c666`
(`f9ea707a5`; Claudexor 4.5.0 shipped from `2b2dfc15e`, whose engine sources are
the same apart from one line of an engine test): `serve`, `both`, `pin`, `no_alternative`
(recognized refusals), `generic_serve`, `generic_both` (an unrecognized HTTP 400),
`third` (two refusing accounts and a serving third), `returns` (one notice per
mark episode, a new one after the mark cleared), `continuation`, `preferred` and
`retry_invalid`. The `older` control is `056c0df9d` (4.4.0 plus #481), tree
`572efe9bf9beac5ba7f697939c476783505afbca`: its catalog does not advertise
`includeCapabilities`, and its refusals carry no retry permission.

`body_utf8` preserves response text with exactly one documented transformation:
at capture time, mechanical prefix replacement maps the driver directory to
`/tmp/cx-capability` and the isolated engine home (both lexical and resolved
spellings) to `/tmp/cx-capability/<scenario>`. No other response bytes change.
Unmodified transcripts stay in the capture run's own `raw/` directory, outside
the repository.

Each distinct operations body is stored once per tree in `operations-*.json`;
other repeated large bodies use `body-*.json`. A wire row's `body_ref` and
`body_sha256` identify that exact text. `read_capture` checks its SHA-256 and
engine tree before replay. `repeat` run-length encodes identical consecutive
polls; replay restores their original count. This changes storage, not the
responses seen by the host. Named guard tests mutate recorded facts explicitly.

`engine.mjs` imports production engine packages. Only Codex auth, provider HTTP
and readiness are injected, following engine tests; the capability ledger reads
real time, skewed forward five hours only in `retry_invalid` before the real
correlation validator. SQL, selection, classification, marks, uploads,
continuations and the HTTP server are real. Global fetch is denied; every run owns
a fresh home, socket and ephemeral port. `returns` steers which accounts serve
through `provider-mode.json` in the run's home. `preferred` injects a soft account
preference into the host's prepared request without account-bound history.

Run via `scripts/safe_test.py` with `CAPABILITY_ENGINE_ROOT` (a built engine
checkout), `CAPABILITY_NODE_BINARY` (Node 24) and optional
`CAPABILITY_CAPTURE_DIR`. The older run also sets
`CAPABILITY_ENGINE_TREE=572efe9bf9beac5ba7f697939c476783505afbca`. The replay
tests need no engine. The browser test serves a byte-verified candidate SPA, the
real Accounts proxy projection and the production notice/history endpoint.
Unrelated Settings APIs use the existing subscription UI fixture.
`CAPABILITY_BROWSER_EXECUTABLE` selects an installed browser; profiles and data
roots remain isolated.
