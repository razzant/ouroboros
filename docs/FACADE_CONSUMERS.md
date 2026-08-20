# Facade consumer inventory

The v7 module splits left every parent module re-exporting what moved out of it, so
existing callers and monkeypatching tests kept working unchanged. A family of pinning
tests already proves those bindings are *the same objects* as the leaves'. What no
pin answers is the question this document answers: **who actually consumes each
retained binding.**

This is the classification layer on top of the identity pins. It does not restate
their name lists — each row cites the pin instead, and the pin remains the authority
on which names a facade must keep.

## Method

Derived mechanically by AST, not by reading the pins. For every module under
`ouroboros/`, `supervisor/` and `server.py`:

1. A **retained binding** is a name introduced by a top-level `from X import ...`
   whose statement carries a `noqa: F401` marker. That marker is how this codebase
   declares "this import exists for its binding, not for this module's own use", and
   it is the only declaration a scanner can see.
2. Consumers are then counted across four populations, resolving dotted module
   aliases and the `_loop()` / `_go()` / `_queue()`-style **call-time parent handles**
   the D18/D33/D35/D36/D37/D38 leaves use, so a leaf reading `_loop().X` counts as a
   consumer of `ouroboros.loop.X`:
   - product code — `ouroboros/`, `supervisor/`, `server.py`;
   - peripheral code — `scripts/`, `devtools/`, `skills/`;
   - the parent module's own body (a bare `Name` load below the import block);
   - tests — `from <facade> import name`, `<facade>.name`, and
     `monkeypatch.setattr` in both its object and dotted-string forms.

**Known limits, stated so the numbers are not over-read.** A `noqa: F401` on a
multi-name import block covers every name in that block, so a block that mixes real
imports with compatibility bindings inflates the retained-binding count; the
*parent-internal* column below is exactly that population, separated rather than
hidden. Re-export surfaces declared only through `__all__`, or through plain imports
with no marker, are invisible to this scan. And a name whose only consumer is a
pinning test that enumerates it as a **string** appears here as contract-only — which
is the honest answer: the pin is the consumer.

## Consumer classes

Each retained binding falls into exactly one primary class:

- **(a) runtime caller** — some other module in the repo reads the name *through the
  facade*. The counted files are its real callers.
- **(parent-internal)** — the facade's own body uses it. Not a compatibility binding
  at all; it is an ordinary import sitting inside a marked block.
- **(test-only)** — no runtime consumer; tests import, read or patch it. This is the
  **patch surface**: the seam a test reaches for when it wants to intercept behaviour
  at its historical address.
- **(c) contract-only** — nothing reads it anywhere. It exists because the split
  promised the binding would survive, and because the v7 ledger marks these rows
  `pending upstream transfer`: removing one now would change an address upstream
  still expects.

The **(b) monkeypatched** column is deliberately *not* exclusive. A name can be both a
live runtime caller path and a patch target — that combination is the most
load-bearing kind of binding there is, and collapsing it into a bucket would hide it.

---

## Summary

| facade (parent) | LOC | re-exports (private) | (a) runtime names / caller files | parent-internal | (b) monkeypatched names / test files | test-only | (c) contract-only | identity pin |
|---|---:|---:|---|---:|---|---:|---:|---|
| `ouroboros/loop.py` | 629 | 146 (88) | 76 / 9 | 29 | 26 / 18 | 0 | 41 | `tests/test_loop_owner_facades.py::test_loop_owner_facades_preserve_identity` + `tests/test_module_handle_extraction.py` |
| `supervisor/events.py` | 270 | 109 (78) | 3 / 3 | 31 | 3 / 3 | 19 | 56 | `tests/test_events_extraction.py::test_events_facade_reexports_every_moved_identity` |
| `ouroboros/tools/control.py` | 411 | 106 (56) | 4 / 4 | 23 | 3 / 4 | 15 | 64 | `tests/test_control_extraction.py::test_control_facade_reexports_every_moved_identity` |
| `ouroboros/extension_loader.py` | 956 | 91 (49) | 12 / 15 | 40 | 7 / 16 | 2 | 37 | `tests/test_extension_loader_extraction.py::test_extension_facade_reexports_every_moved_identity` |
| `ouroboros/tools/git.py` | 991 | 88 (56) | 6 / 5 | 20 | 9 / 6 | 8 | 54 | `tests/test_git_extraction.py::test_git_facade_reexports_every_moved_identity` |
| `supervisor/queue.py` | 430 | 88 (25) | 45 / 27 | 6 | 27 / 41 | 11 | 26 | `tests/test_module_handle_extraction.py::test_the_queue_facade_still_exports_everything_that_moved` + `tests/test_cancel_custody_extraction.py::test_task_lifecycle_facade_reexports_every_moved_identity` |
| `ouroboros/tools/scope_review.py` | 891 | 83 (57) | 4 / 2 | 27 | 3 / 6 | 11 | 41 | `tests/test_scope_review_extraction.py::test_scope_review_facade_reexports_every_moved_identity` |
| `ouroboros/tools/shell.py` | 720 | 80 (48) | 11 / 13 | 19 | 4 / 8 | 4 | 46 | `tests/test_shell_extraction.py::test_shell_facade_reexports_every_moved_identity` |
| `ouroboros/config.py` | 900 | 79 (10) | 54 / 84 | 8 | 6 / 10 | 4 | 13 | `tests/test_config_extraction.py::test_config_facade_reexports_every_moved_identity` |
| `ouroboros/llm.py` | 716 | 64 (36) | 6 / 8 | 18 | 1 / 1 | 10 | 30 | `tests/test_llm_extraction.py::test_llm_facade_reexports_every_moved_module_identity` |
| `ouroboros/skill_review.py` | 688 | 60 (34) | 7 / 4 | 15 | 1 / 1 | 9 | 29 | `tests/test_skill_review_extraction.py::test_skill_review_facade_reexports_every_moved_identity` |
| `ouroboros/review_substrate.py` | 834 | 58 (19) | 17 / 9 | 12 | 2 / 1 | 8 | 21 | `tests/test_review_substrate_extraction.py::test_review_substrate_facade_reexports_every_moved_identity` |
| `ouroboros/tools/claude_advisory_review.py` | 909 | 58 (31) | 11 / 6 | 10 | 9 / 6 | 3 | 34 | `tests/test_review_owner_facades.py::test_review_owner_facades_preserve_identity` + `tests/test_module_handle_extraction.py` |
| `ouroboros/tools/review_helpers.py` | 771 | 55 (20) | 42 / 24 | 2 | 0 / 0 | 1 | 10 | `tests/test_review_helpers_extraction.py::test_review_helpers_facade_reexports_every_moved_identity` |
| `supervisor/workers.py` | 724 | 55 (28) | 18 / 11 | 13 | 10 / 21 | 5 | 19 | `tests/test_worker_process_extraction.py::test_workers_facade_reexports_every_moved_identity` + `tests/test_module_handle_extraction.py` |
| `ouroboros/tools/delegate.py` | 1270 | 53 (49) | 2 / 2 | 20 | 2 / 2 | 8 | 23 | `tests/test_delegate_owner_facades.py::test_delegate_owner_facades_preserve_identity` + `tests/test_module_handle_extraction.py` |
| `ouroboros/headless.py` | 903 | 52 (31) | 11 / 15 | 10 | 1 / 4 | 2 | 29 | `tests/test_headless_extraction.py::test_headless_facade_reexports_every_moved_identity` |
| `ouroboros/tool_access.py` | 734 | 50 (17) | 25 / 52 | 12 | 2 / 2 | 2 | 11 | `tests/test_tool_access_extraction.py::test_tool_access_facade_reexports_every_moved_identity` |
| `server.py` | 1421 | 49 (45) | 0 / 0 | 23 | 5 / 5 | 15 | 11 | `tests/test_server_extraction.py::test_server_facade_reexports_every_moved_identity` |
| `ouroboros/review_state.py` | 659 | 47 (39) | 7 / 6 | 18 | 0 / 0 | 0 | 22 | `tests/test_review_state_extraction.py::test_review_state_facade_reexports_every_moved_identity` |
| `ouroboros/tools/review.py` | 1245 | 41 (13) | 5 / 2 | 24 | 9 / 9 | 4 | 8 | `tests/test_review_owner_facades.py::test_review_owner_facades_preserve_identity` + `tests/test_module_handle_extraction.py` |
| `supervisor/task_lifecycle.py` | 765 | 40 (27) | 16 / 6 | 4 | 1 / 1 | 1 | 19 | `tests/test_cancel_custody_extraction.py::test_task_lifecycle_facade_reexports_every_moved_identity` |
| `ouroboros/usage_accounting.py` | 1377 | 36 (26) | 5 / 18 | 10 | 6 / 7 | 7 | 14 | `tests/test_lc2_owner_facades.py::test_lc2_owner_facades_preserve_identity` + `tests/test_module_handle_extraction.py` |
| `supervisor/git_ops.py` | 605 | 35 (12) | 35 / 16 | 0 | 18 / 9 | 0 | 0 | `tests/test_git_ops_owner_facades.py::test_git_ops_owner_facade_preserves_identity` + `tests/test_module_handle_extraction.py` |
| `ouroboros/tools/registry.py` | 39 | 32 (26) | 8 / 86 | 0 | 1 / 2 | 7 | 17 | `tests/test_tool_owner_facades.py::test_tool_descriptor_owner_facades_preserve_identity` + `tests/test_registry_core.py::test_registry_core_extraction_preserves_only_proven_facades` |
| `ouroboros/review_evidence.py` | 826 | 28 (24) | 3 / 5 | 16 | 1 / 2 | 2 | 7 | `tests/test_review_evidence_extraction.py::test_review_evidence_facade_reexports_every_moved_identity` |
| `ouroboros/tools/plan_review.py` | 963 | 28 (28) | 0 / 0 | 22 | 4 / 5 | 3 | 3 | — |
| `ouroboros/agent.py` | 1137 | 26 (9) | 0 / 0 | 15 | 0 / 0 | 1 | 10 | `tests/test_lc2_owner_facades.py::test_lc2_owner_facades_preserve_identity` + `tests/test_module_handle_extraction.py` |
| `ouroboros/agent_task_pipeline.py` | 1111 | 26 (16) | 0 / 0 | 17 | 7 / 5 | 2 | 7 | `tests/test_lc2_owner_facades.py::test_lc2_owner_facades_preserve_identity` |
| `ouroboros/review_execution.py` | 1370 | 19 (10) | 3 / 1 | 7 | 0 / 0 | 4 | 5 | `tests/test_review_owner_facades.py::test_review_owner_facades_preserve_identity` |
| `ouroboros/tools/subagent_integration.py` | 1030 | 13 (11) | 6 / 2 | 2 | 2 / 2 | 0 | 5 | `tests/test_delegate_owner_facades.py::test_delegate_owner_facades_preserve_identity` + `tests/test_module_handle_extraction.py` |
| `ouroboros/delegate_custody.py` | 1275 | 11 (5) | 7 / 7 | 0 | 7 / 7 | 1 | 3 | `tests/test_delegate_owner_facades.py::test_delegate_owner_facades_preserve_identity` + `tests/test_module_handle_extraction.py` |
| `ouroboros/tools/delegate_integration.py` | 871 | 7 (6) | 1 / 1 | 1 | 0 / 0 | 0 | 5 | `tests/test_delegate_owner_facades.py::test_delegate_owner_facades_preserve_identity` + `tests/test_module_handle_extraction.py` |
| `ouroboros/gateway/tasks.py` | 1456 | 5 (2) | 3 / 2 | 0 | 1 / 1 | 2 | 0 | — |
| `ouroboros/context.py` | 1318 | 4 (4) | 0 / 0 | 4 | 0 / 0 | 0 | 0 | — |
| `ouroboros/subagents.py` | 1382 | 4 (3) | 1 / 3 | 0 | 1 / 3 | 1 | 2 | — |
| `supervisor/update_merge.py` | 1202 | 4 (2) | 2 / 1 | 0 | 2 / 2 | 0 | 2 | `tests/test_update_merge_owner_facade.py::test_update_merge_owner_facade_preserves_identity` + `tests/test_module_handle_extraction.py` |
| `ouroboros/provider_models.py` | 478 | 3 (0) | 1 / 2 | 1 | 0 / 0 | 0 | 1 | — |
| `ouroboros/contracts/api_v1.py` | 12 | 1 (1) | 1 / 1 | 0 | 0 / 0 | 0 | 0 | — |
| `ouroboros/review.py` | 1099 | 1 (0) | 0 / 0 | 0 | 0 / 0 | 1 | 0 | — |
| `ouroboros/review_cycles.py` | 159 | 1 (0) | 1 / 3 | 0 | 0 / 0 | 0 | 0 | — |
| `supervisor/events_evolution_done.py` | 153 | 1 (1) | 0 / 0 | 1 | 0 / 0 | 0 | 0 | — |

**Totals:** 42 facade modules, 1837 retained bindings — 459 runtime, 480
parent-internal, 173 test-only, **725 contract-only**, with 181 bindings monkeypatched
somewhere in the suite.

---

## Per-facade notes

### `ouroboros/loop.py` — 146 bindings, 88 private

The reference case for the whole family. Its runtime consumers are **exclusively nine
of its own leaves** (`loop_acceptance`, `loop_acceptance_review`, `loop_budget`,
`loop_delivery`, `loop_forced_finalization`, `loop_messages`, `loop_model_call`,
`loop_nudges`, `loop_round_limits`), every one of them reading through the D33
call-time handle `_loop()`; the other two, `loop_llm_call` and `loop_tool_execution`,
read nothing back through the parent. Nothing outside the loop family imports a
private name from it. That is the point the pin's docstring makes and this scan confirms: a
surviving private re-export exists either because `run_llm_loop`'s own body calls it,
or because a *sibling leaf* reads it through the rendezvous binding — and retiring
those would replace one shared seam with a mesh of sibling handles.

Twenty-six of its bindings are monkeypatched across 18 test files, which is why the
handle exists at all: a test that rebinds `loop.call_llm_with_retry` must still
intercept a body that now lives in `loop_llm_call`.

The 41 contract-only bindings are almost entirely the declared "historical import
surface" — `dataclass`, `field`, `replace`, `estimate_tokens`, `add_usage`,
`extract_final_answer`, the `ACCEPTANCE_*` and `REASON_*` vocabularies. The comment on
those imports says they are kept "for the L-B leaves", and for several the leaves in
fact import from the true owner instead: `loop_nudges` takes `estimate_tokens` from
`ouroboros.utils` directly, so `ouroboros.loop.estimate_tokens` has no reader at all.
That is a bookkeeping observation, not a defect — the binding still costs nothing but
an import — and it is exactly the population the pin calls TEMPORARY (spec 4.3-15).

Pins: `tests/test_loop_owner_facades.py` splits the surviving set from
`RETIRED_FROM_LOOP` and asserts the absence of the retired names, so a well-meaning
re-export cannot silently resurrect a second address.

### `ouroboros/tools/registry.py` — 39 lines, 32 bindings, 26 private

The smallest facade and the most-consumed. Three descriptor names carry almost all of
it — `ToolContext` (70 files), `ToolEntry` (37), `ToolRegistry` (16) — which makes this
39-line module one of the widest import surfaces in the tree, **86 files** in total.

The split between public and private is not clean, and that is the finding. Three
*private* bindings also have live runtime consumers —
`_authorized_managed_update_resolver` (6 files), `_builtin_tool_availability` and
`_compose_execute_result` — while `BrowserState`, a public name, is read only by
tests. The remaining 17 private bindings have no consumer at all: their only reader is
`tests/test_registry_core.py::test_registry_core_extraction_preserves_only_proven_facades`,
which enumerates them as strings.

So the halves warrant opposite treatment, but the line falls on measured consumers
rather than on the leading underscore: a real dependency hub that must not move, a
handful of private names doing real work, and a contract tail retained because the
split promised it and the ledger marks the rows pending upstream transfer.

Leaves: `tool_resolution` (16), `registry_guards` (11), `tool_context` (2),
`tool_catalog`, `registry_core`, `tool_result`.

### `supervisor/git_ops.py` — 35 bindings, zero contract-only

The cleanest facade in the inventory: **every** retained binding has a live runtime
consumer, and 18 of the 35 are also monkeypatched across nine test files. Sixteen
product files reach through it, including `ouroboros/tools/git.py`,
`ouroboros/gateway/control.py`, `supervisor/update_merge.py` and its own four leaves
(`git_ops_remotes`, `git_ops_updates`, `git_ops_reset`, `git_ops_rescue`) via the D35
handle `_go()`.

This is the shape the family is aiming at: the facade is not compatibility ballast, it
is the module's rebindable state surface. `init` rebinds `REPO_DIR` / `DRIVE_ROOT` /
`BRANCH_*` on the parent, so the leaves *must* read through it. `utc_now_iso` is
re-exported for precisely that reason and is read as a `git_ops` attribute by
`supervisor/update_recovery.py`.

### `supervisor/update_merge.py` — 4 bindings

The narrowest split: one leaf (`update_merge_plan`), four names, one runtime consumer
(`ouroboros/gateway/control.py`), two monkeypatched. `_git_run` and
`_build_clean_merge_commit` are contract-only. Note that `update_merge` reaches
`git_ops` the other way — through the module object `_g`, so a test patching
`git_ops.REPO_DIR` is followed by these primitives.

### `ouroboros/tools/control.py` — 106 bindings, 64 contract-only

The largest contract-only population in the tree, and the sharpest example of the
class. Only **four** product files read anything through it
(`review_evidence_sections`, `control_delegation`, `tools/delegate`,
`tools/join_ledger`); its seven `control_*` leaves own the bodies; and the remaining 64
bindings — the whole `_attach_swarm_intent` / `_wait_for_routing_annotation` /
`_prepare_child_drive` family, plus re-exported utilities like `load_settings`,
`save_settings`, `append_jsonl`, `sha256`, `run_cmd`, `Path` and `Any` — have no
reader anywhere.

Unlike `loop.py`, this facade's `noqa` markers were applied to the whole historical
import block, so a large part of the count is stdlib and cross-module utility names
that were never a compatibility promise. It is the clearest candidate for a future
narrowing pass, and the reason the pins call the private half temporary.

### `ouroboros/agent.py` — 26 bindings, zero runtime consumers

Nothing in the repo imports anything from `ouroboros.agent`; it is the top of its
dependency cone. Fifteen bindings are used by its own body (the `agent_dispatch`
leaf's names, which `agent.py` calls directly), one is test-imported, and ten are
contract-only — `CapabilityDelta`, `EFFORT_SCALE`, `SubagentExecutorResolution`,
`resolve_effort`, `envelope_from_task` and friends, re-exported from
`ouroboros.subagents` and `ouroboros.config`.

The L-C2 pin covers `agent.py` together with `agent_task_pipeline.py` and
`usage_accounting.py`; `agent_task_pipeline` is the one of the three with a real patch
surface (7 names across 5 test files, the post-task synthesis seams).

### `ouroboros/llm.py` — 64 bindings, mixin host

Structurally different from the rest: `llm.py` is not only a re-export surface, it is
the **composition point**. `LLMClient` is defined here with ten mixin bases pulled
from the lane leaves — `_PayloadCachePolicyMixin`, `_CapabilityPolicyMixin`,
`_ProviderRoutingMixin`, `_MessageShapingMixin`, `_RecoveryLadderMixin`,
`_AnthropicLaneMixin`, `_GigaChatLaneMixin`, `_LocalLaneMixin`,
`_OpenAICompatibleLaneMixin`, `_GenerationCostMixin` — so the class exists nowhere
else and the leaves are not independently instantiable.

Consequently its re-exports split cleanly: 18 are the mixins and helpers the class
body itself needs, 6 have runtime consumers through the facade (`loop_llm_call`,
`context_compaction`, `pricing`, `vision_routing`, the two `control_*` result
surfaces), 10 are test-only, and 30 are contract-only — mostly `llm_attempt`'s
physical-attempt vocabulary (`execute_physical_attempt`, `current_usage_scope`,
`AttemptRequest`) and the private compaction/refusal predicates. The leaves
deliberately do **not** read back through the facade;
`ouroboros/llm_probe.py` states the rule in place: name the owner leaf, never the
`llm.py` facade.

The pin additionally asserts that every `LLMClient` member resolves to its mixin owner
and that the member inventory is unchanged — a stronger contract than binding identity
alone.

### Two facades that are dependency hubs, not compatibility shims

`ouroboros/config.py` (54 runtime bindings across **84** files) and
`ouroboros/tool_access.py` (25 across 52) have the lowest contract-only ratios of any
large facade. `ouroboros/tools/review_helpers.py` is the same shape (42 across 24).
For these, "facade" is the wrong mental model: the re-exports are the module's public
API, and the leaves below them are implementation detail. Only ten of `config.py`'s 79
bindings are private at all, and only 13 have no reader anywhere — the lowest
contract-only share in the table.

### Two facades whose consumers are almost entirely tests

`supervisor/events.py` (109 bindings, 3 runtime, 56 contract-only) and
`server.py` (49 bindings, 0 runtime, 11 contract-only) are consumed overwhelmingly by
the test suite. `events.py` is a dispatcher: the real work lives in ten `events_*`
leaves that never import the dispatcher they serve, so the only runtime readers are
`server_owner_routing`, `steering` and `task_reaper`. `server.py` is a composition
root whose retained facade bindings no production code reads — the one runtime
importer is `ouroboros/cli.py`, which imports the module to call `server.main()`
and touches none of the 49 bindings; they exist so `server.<name>` keeps resolving
for the 15 test-only and 5 monkeypatched names that reach into it.

### `supervisor/queue.py` — the widest patch surface

27 monkeypatched bindings across **41** test files, the largest of any facade, on top
of 45 runtime bindings read by 27 product modules. This is the module the D18
mechanism was invented for: `init_queue_refs` rebinds `PENDING`, `RUNNING`,
`DRIVE_ROOT` and the rest, dozens of test sites rebind them on the parent, and a leaf
holding a from-import would freeze the object it saw at import time. Its four leaves
read `_queue()` at call time for exactly that reason.

### `ouroboros/tools/plan_review.py` — 28 bindings, no identity pin

The one sizeable facade with **no** `*_facade_reexports_every_moved_identity` pin. All
28 bindings are private, 22 are used by the module's own body, 4 are monkeypatched
across 5 test files, and 3 are contract-only. Its re-exports are aliased on import
(`X as _X`), so the historical private spelling is preserved without the leaf having
to own the underscore. Peripheral consumers exist outside the scanned import shape:
`scripts/run_plan_review.py` imports `_PlanRequest` and `_run_plan_review_async`
directly. Absence of a pin is recorded here as a fact, not a recommendation — adding
one would be a code change, and this document is evidence.

---

## Cross-cutting observations

1. **Contract-only is the majority class.** 725 of 1837 retained bindings (39%) have
   no reader in product code, peripheral code, the parent body, or the tests. Nine
   facades account for most of them: `control.py` (64), `events.py` (56), `git.py`
   (54), `shell.py` (46), `loop.py` (41), `scope_review.py` (41),
   `extension_loader.py` (37), `claude_advisory_review.py` (34), `llm.py` (30).
2. **The two extreme shapes are worth naming.** `supervisor/git_ops.py` has zero
   contract-only bindings and half its surface monkeypatched — a facade doing real
   work. `ouroboros/tools/control.py` has 64 contract-only against 4 runtime callers —
   a facade that is almost entirely promise. Both passed the same pin.
3. **A facade's size in lines says nothing about its consumer load.**
   `ouroboros/tools/registry.py` is 39 lines and reaches 86 files;
   `ouroboros/gateway/tasks.py` is 1456 lines and re-exports 5 names to 2 files.
4. **Test patch surface concentrates in the supervisor.** `queue.py` (41 test files),
   `workers.py` (21), `loop.py` (18), `extension_loader.py` (16) — together more than
   half the 181 patched bindings. These are the seams a narrowing pass would break
   first, and the reason the module-handle mechanism was approved as an exception
   rather than the leaves being allowed to own their own copies.
5. **Peripheral consumers exist and are easy to miss.** `devtools/benchmarks/**` reads
   through `ouroboros/config.py`, `ouroboros/extension_loader.py` and
   `ouroboros/tools/registry.py`; `scripts/run_external_review.py` reads through
   `ouroboros/config.py` and `ouroboros/tools/git.py`; `scripts/v7_evidence.py` reads
   through `ouroboros/extension_loader.py`. A retirement judged on `ouroboros/` and
   `supervisor/` alone would break these.

## What this inventory does not decide

Nothing here is a retirement list. A contract-only binding is not dead code: the v7
ledger marks these rows `pending upstream transfer`, and the addresses are what an
upstream merge resolves against. Retiring any of them is a ledger decision with its
own row and its own pin update — the same shape as `RETIRED_FROM_LOOP`, where the
*absence* of the binding became the asserted contract. This document exists so that
decision can be made against measured consumers instead of a guess.
