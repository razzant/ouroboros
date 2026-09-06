# F3.1 lane A design note — the typed tool-result organ (D02 re-derivation)

Design-note-before-code (plan §5.4 rule). Audience: the F3.1 lane A operator.
Base for every claim: `ouroboros_v7next @ db944347`; oracle: `v7_wip @ 9f691656`
(frozen). Everything below is RE-DERIVED against tip bytes — verbatim reuse of
oracle spans is forbidden (re-prove trap, ledger D15 entry 3).

## Why this lane is first in the F3.1 fan-out

ABI-4 (`ResolvedModelTarget`, D02-owner) and ABI-6(б) (`_typed_or_adapted`
branch — exists ONLY in the oracle's `loop_tool_execution.py`, zero tip hits)
both execute inside this re-derivation; the lane's protection-closure returns
the D04 remainder (registry_core/tool_result into
SAFETY_CRITICAL_PATHS/HOT_CODE_PATHS).

## Tip facts (spot-verified on db944347)

- Zero `ToolResult` occurrences on the tree; `tools/registry.py` = 2686 lines
  (ToolRegistry class ~2252 of them, from line 435); `loop_tool_execution.py`
  = 1390 lines.
- Oracle organs: `tools/tool_result.py` (961 lines, 33 symbols),
  `tools/registry_core.py` (1139); suites `test_tool_result{,_meta_boundaries,_t46}.py`,
  `test_registry_core.py`, `test_tool_classification_differential.py`,
  `test_tool_execution_classification.py`, `tool_classification_corpus.py`,
  `test_llm_typed_policy_refusal.py`.

## Composition (HOT-DEFERRED ledger rows; all re-derive, see f3 plan §4)

1. `registry_core.py` — D04 entry 3: rows 156/167/170/171/174/175 + 17
   method→function extractions (receiver `self`→`registry`); the class does
   not fit the band whole (>1500) — decompose through the extractions (Q11=B);
   python_interpreter/artifacts import bindings (rows 213/214, 246/247) ride
   along.
2. `tool_result.py` — D04 entry 4: closed code table, ToolResult/ToolCodeSpec;
   row 139 `_compose_execute_result` is drifted — take tip bytes.
3. `extension_dispatch` typed dispatchers — D04 entry 5 (rows 187/188 +177
   producer-boundary lines) + `failure_kind` from extension_process_runner
   (D14 entry 10). The ABI-9 digest READ in this file is the F3.2 seam, not
   this lane.
4. `loop_tool_execution` cutover — rows 157-164, 826-828: retire result-text
   classification (the «D02-петля» mandatory return). ABI-6(б) resolves here:
   the unreachable `_typed_or_adapted` branch is NOT reproduced.
5. `_outcome_tool_errors` T1-partition + `reflection._trace_call_errored`
   (D15 entries 3-4; re-derive against upstream status handling — the naive
   port INVERTS the fix) + row 166 (retire 4 CLAUDE_CODE markers, 0 emitters).
6. D09 typed-policy-refusal subfamily — D02 entry 4: rows
   1706/1749/1751/1759/1760 + PROVIDER_POLICY_REFUSAL machinery in
   llm_attempt, classification in loop_llm_call; pins
   `test_llm_typed_policy_refusal.py` + fallback_ladder.json goldens 17→15.
7. Producer cutovers (tip==merge-base, reference typed):
   core_file_tools/core_artifacts (10 producers via `_publish_tool_result`,
   incl. row 332), shell_outputs 3-tuple, services.py, mcp_client,
   tools/git a5e1cea3-cutover (`_publish_git_error`/`_publish_review_blocked`
   + typed `_git_status`/`_git_diff`/stage cycles), control rows
   2548/2549/2556/2571/2574/2579 on the F2.1 leaves.
8. Test rows 832-833 + non-carried D04 entry 9 pins + protection-closure
   (SAFETY_CRITICAL_PATHS/HOT_CODE_PATHS return, D04 entry 11).

## Boundaries

- Do NOT touch the ~60-70 str-returning tool handlers: handler-ABI conversion
  is ABI-8 = POST-RELEASE (owner Q5=A + Q16=A; validator pins phase=POST).
  Exactly one LegacyTextResultAdapter remains, with an inventory — the
  owner-approved residual.
- ADOPTION hook for D02: `tests/test_tool_classification_differential.py` +
  `tests/test_tool_result.py` (suites arrive with this lane).
- Size law Q11=B (1600 hard / band-rationale); `-m size_ratchet` before every
  integration hand-off; ARCHITECTURE.md delta rides the same commit.
