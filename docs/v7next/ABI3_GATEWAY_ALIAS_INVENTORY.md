# ABI-3 — F11 per-alias inventory (gateway compat aliases, ABI 7.0 window)

Frozen BEFORE the first removal (lane D3, base 29e2b045). Axes per F11:
ingress / egress / JS / producer / stored / migration / removal-test.
This document is the ABI-3 feeder inventory for the RC auditor scope
(docs/v7next/DESIGN_RC_AUDIT_SCOPE.md); the removal tests live in
`tests/test_gateway_abi3_removals.py`.

Lane constraint (coordinator, 2026-08-31): `web/` JS is untouchable in this
lane (`chat.js` sits at its BYTE_DEBT ceiling). Verified consequence: NO alias
requires a functional web-client edit for its removal — the only JS artifacts
are stale JSDoc typedef lines and the `GATEWAY_CONTRACT_VERSION` carrier
switch in `web/modules/api_types.js`, both HOT-DEFERRED (evidence per alias
below), so no alias itself is deferred.

## 1–2. `cost_usd` / `cost_usd_with_children` (ChatOutbound)

- Declaration: `gateway/contracts.py` ChatOutbound (deprecated aliases beside
  the honest `accounted_upper_bound_usd[_with_children]` names, C2/owner 10=B).
- ingress: none — outbound-only fields; no inbound surface declares them.
- egress: WS chat frames (task_done / heartbeat / progress / subagent meta)
  exclusively through the cost SSOT emitters
  `ouroboros/cost_projection.py::{with_cost_aliases, carry_cost_meta,
  cost_projection, live_root_cost_projection}`; task-result open-shape
  passthrough on `GET /api/tasks/{task_id}` for records stamped with the
  alias; history replay of stored `chat.jsonl` rows via
  `task_results.TASK_COST_META_FIELDS`.
- JS: `web/modules/utils.js` `resolveCostPair` resolves the pair with
  deprecated-wins precedence AND an honest-name fallback. Removal DID need a
  web edit, made in the stage-2 fix wave: `chat.js::costMetaKeys` copies the
  retired names unconditionally, and nothing serializes them away on that
  in-memory path (no JSON round-trip runs between the whitelist and the
  reader), so an alias-free frame reached the reader with `cost_usd*` as own
  properties valued `undefined`. The resolver read that as "the deprecated
  name is present" and answered null, freezing a subagent card on
  "cost pending" for the rest of the run. Presence there now means a DEFINED
  value; an explicit `null` is still present (Python parity with `old in src`)
  and a mirrored legacy amount still wins its pair. The JSDoc typedefs already
  name the honest fields: `accounted_upper_bound_usd` and
  `accounted_upper_bound_usd_with_children` at `api_types.js:287` and `:290`
  on this tree (this sentence was written in the stage-2 fix wave, so unlike
  the frozen-base citations elsewhere in this file it names current lines;
  `:288/:291` were the two description lines under them).
- producer: `cost_projection.py` is the one author of the emitted pair;
  `gateway/tasks.py:227` stamps `cost_usd=0.0` into the stored
  admission-failure record; hand literals (`agent_task_pipeline.py`,
  `supervisor/events_task_done.py:256`) flow through the SSOT seam.
- stored: `task_results/<id>.json`, `chat.jsonl` rows, and task_summary rows
  carry historical alias keys. Read tolerance is KEPT: `resolve_cost_pair`
  still reads both spellings (deprecated wins on a diverged stored pair) and
  `TASK_COST_META_FIELDS` still carries stored legacy keys through history
  replay. Replay of history is never rejected (stored axis, per plan).
- migration: none required — the additive honest names shipped in C2 and both
  Python and JS readers already resolve the pair.
- removal: SSOT emitters stop emitting the deprecated spellings and STRIP them
  from write-side copies (`with_cost_aliases` normalizes a producer literal
  away); contracts.py drops the two ChatOutbound fields;
  `gateway/tasks.py:227` stamps the honest name. Live-frame readers that
  consumed the emitted alias key (`agent_task_pipeline.py:842/:853`) switch to
  the honest name.
- removal-test: declaration gone from ChatOutbound hints; emitters emit no
  alias key; deprecated-wins read tolerance for stored pairs pinned.

## 3. `telegram_chat_id` (ChatOutbound / PhotoOutbound / VideoOutbound / DocumentOutbound)

- Declaration: 4 `NotRequired[int]` fields in `gateway/contracts.py`
  (deprecated compatibility twins; runtime emits `transport` instead).
- ingress: none — never declared on any inbound surface.
- egress: ONLY the history replay mapper `gateway/history.py` —
  `:849` re-emits `int(entry.get("telegram_chat_id") or 0)` unconditionally
  (every replayed frame carried `telegram_chat_id: 0`), `:381`/`:401`
  hard-code `0` into synthesized origin rows. No live producer emits it.
- JS: JSDoc typedefs only (`api_types.js:333/:370/:389/:410`); zero
  functional readers in `web/modules` — no web edit needed; typedef cleanup
  HOT-DEFERRED (web/ untouchable).
- producer: none live (no runtime writer of the key into chat.jsonl since the
  transport generalization).
- stored: historical `chat.jsonl` rows carry the key. Readers TOLERATE it
  (the mapper reads-and-ignores after removal; replay is never rejected, and
  ingress validation does not apply to replay — it is an egress path).
- migration: `transport` (`TransportMetadata`) shipped earlier as the
  replacement.
- removal-test: declaration gone from all four outbound TypedDicts; a legacy
  stored row WITH the key still replays (without re-emitting it); synthesized
  rows carry no dead zero field.

## 4–5. `project_last_viewed` / `project_hidden` (UiPreferencesResponse)

- Declaration: `gateway/contracts.py` UiPreferencesResponse (marked
  "deprecated one-minor accepted no-op" — the one-minor window has closed).
- ingress: `POST /api/ui/preferences` accepted them as LOUD no-ops
  (`deprecated_ui_preferences_ignored` warning + zeroing). After removal the
  existing unknown-key policy answers 400 — the documented end state of a
  one-minor acceptance window.
- egress: always emitted as `{}` defaults on GET/POST responses.
- JS: JSDoc typedef only (`api_types.js:1017-1018`); the shipped client
  neither reads nor sends the keys (it writes `project_seen_revision`) — no
  web edit needed; typedef cleanup HOT-DEFERRED (web/ untouchable).
- producer: `gateway/ui_preferences.py` (defaults, normalize branch, POST
  zeroing, `_legacy_keys`/`_deprecated_warning` machinery).
- stored: `state/ui_preferences.json` may carry legacy values; after removal
  unknown stored keys are ignored on read by `_normalize_preferences`
  construction (only known keys propagate) and dropped on the next write —
  tolerated, never fatal.
- migration: `project_seen_revision` (shipped replacement).
- removal-test: fields gone from the response contract and payloads; POST
  with a legacy key → 400 unknown-key; a stored legacy file still loads.

## HOT-DEFERRED (evidence, not removals)

- `web/modules/api_types.js` stale JSDoc typedef lines for all five aliases
  (`:283`, `:291`, `:333`, `:370`, `:389`, `:410`, `:1017-1018`): comment-only
  drift, zero runtime effect; web/ JS untouchable in this lane.
- `GATEWAY_CONTRACT_VERSION` in `api_types.js` (currently `'6.113.4'`) as the
  JS-side ABI-version mirror: the server-side carrier
  (`gateway.schema.GATEWAY_ABI_VERSION`) lands in this lane; the JS mirror
  switch + sync pin follow when a web lane opens.
