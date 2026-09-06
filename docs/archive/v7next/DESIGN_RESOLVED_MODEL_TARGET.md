# F3.2 seam design note — ResolvedModelTarget (ABI-4)

Greenfield §6-design: zero occurrences on tip and in the oracle — this is NOT
a transplant. Owner decision: plan §6 item 4 (frozen dataclass, typed
consumption by every lane). Home: the D02-owner domain — the
`model_slots.py` / `provider_models.py` seam (settings vocabulary side), so
the typed organ (lane A) must land first.

## Contract

One frozen dataclass describing a fully RESOLVED model destination — the
output of route resolution, consumed downstream without re-parsing strings:

```python
@dataclasses.dataclass(frozen=True, slots=True)
class ResolvedModelTarget:
    model_id: str          # exact provider model id, e.g. "anthropic/claude-..."
    provider_route: str    # resolved transport lane, e.g. "openrouter" | "openai-compatible" | "local"
    credential_ref: str    # which configured credential/profile serves the call ("" = default)
    effort: str            # normalized reasoning-effort label ("" when N/A)
    context_window: int    # tokens; 0 = unknown (fail-open per cost-unknown rule)
```

Rules:

- Frozen + slots; equality/hash by value. No Optional-by-default sprawl:
  absent facts are typed sentinels ("" / 0), never None-vs-missing ambiguity.
- Constructed ONLY at the existing resolution seams; downstream code takes
  the dataclass, never a raw comma/at-string. No parallel resolver: the
  dataclass wraps what the current resolution already computes (reuse-first).
- No pricing fields: cost stays with the provider-route pricing SSOT
  (hardcoded price tables remain banned).

## Consumers (the F3.2 sweep, after lanes A and D4 integrate)

1. `llm_fallback` candidate ladder — candidates become
   `tuple[ResolvedModelTarget, ...]`.
2. `review_model_routes` / `reviewer_slot_config` — AFTER ABI-10 lands
   (comma-list migration-read removed; slots are the only source).
3. Delegation lanes (delegate/claudexor route pinning) — typed target in the
   run request instead of string slugs re-parsed per adapter.

## Verification hook

`tests/test_resolved_model_target.py` (new; the suite name is fixed by this
note — update the ADOPTION ABI-4 row's hook when the suite lands): frozen-ness,
value identity, construction at each seam, and a consumer sweep pin (grep-level:
no new comma/at-string parsing beside a seam that already yields the dataclass).
