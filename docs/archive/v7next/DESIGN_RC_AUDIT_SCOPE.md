# F3.3 design note — RC auditor machine-readable scope (ABI-7b, F13)

The RC auditor is the migration-window instrument of Q6=A: a command that
scans a THIRD-PARTY install (skill manifests + settings document) and names
every ABI-7.0 incompatibility with its migration, before the owner upgrades.
It runs LAST in F3 (serial tail): its scope is the UNION of the FROZEN final
inventories of every F3 lane, so it cannot be built before they land.

## Scope schema (machine-readable, one JSON document)

```json
{
  "abi": "7.0",
  "sources": {"tree": "<sha>", "inventories_frozen_at": "<sha>"},
  "checks": [
    {"id": "gateway-alias", "surface": "...", "removed": "...", "replacement": "...", "migration": "..."},
    {"id": "retired-setting", "key": "...", "since": "7.0", "behavior": "stripped-on-load", "migration": "..."},
    {"id": "comma-list", "key": "...", "replacement": "reviewer slots", "migration": "move config to slots BEFORE upgrade"},
    {"id": "plugin-api", "requirement": "manifest plugin_api field", "grandfather": "hash-bound PASS", "migration": "..."},
    {"id": "schema-stamp", "entity": "task_results", "consequence": "pre-7.0 history quarantined (Q8=B, BY DESIGN)"}
  ]
}
```

Feeder inventories (each lane freezes its list as data, not prose):

- ABI-3: the per-alias inventory (F11 axes: ingress/egress/JS/producer/
  stored/migration/removal-test) — five gateway aliases.
- ABI-5: the Q10-retired keys (`OUROBOROS_SCOPE_REVIEW_FLOOR` in
  `RETIRED_SETTING_KEYS`; removed knobs `until_deadline`,
  `stall_rounds_threshold`; removed `fail_tasks` has no install-visible key —
  it is named only in the report prose).
- ABI-10: comma-list keys retired to `RETIRED_SETTING_KEYS`
  (exact list snapped from `settings_defaults.py` at execution time).
- ABI-1: plugin_api admission facts (absent field ≡ LEGACY "1.3";
  new-PASS admission predicate; hash-bound grandfather).
- ABI-2: `_schema_version=1` stamps; the auditor MUST name the Q8=B
  consequence: pre-7.0 task-result history is quarantined after upgrade,
  deliberately (no converter exists).

## Behavior

- Read-only over the audited install; never mutates it. Output: typed report
  (JSON + human rendering), exit 0 = clean, 1 = incompatibilities found,
  2 = install unreadable or the audit itself failed (traversal/report-write
  OSError; PYTHONPYCACHEPREFIX inside the audited root without startup
  bytecode suppression). A mandatory source the audit cannot read/parse is a
  BLOCKING `unauditable-source` finding (exit 1, an audit-integrity plane
  outside the five scope classes) — never a silent exit 0.
- N−1 fixtures (F14, shared with ABI-2/ABI-7a): a settings document and a
  skill manifest authored by the previous minor run through the auditor as
  test fixtures — real bytes, not synthetic shapes.
- Everything not machine-checkable stays an owner-attestation LIST the
  auditor prints (F13 decision) — no pretend-coverage.

## Verification hook

RC audit fixture suite (new, F13/F14) — named in the ADOPTION ABI-7 row;
the auditor script lands under `scripts/` in F3.3.
