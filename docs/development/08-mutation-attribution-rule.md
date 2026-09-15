# Mutation Attribution Rule

This chapter owns the rule that attribution is evidence rather than exclusion: the host captures a baseline when a queued root starts, and blockers ride into review and acceptance evidence instead of becoming structural outcome vetoes. It also fixes what a reviewed commit may stage and where an unversioned interpreter may be resolved, because both decide whose changes a commit actually carries.

- Attribution is evidence, not exclusion: the host captures a `system_repo`
  baseline when a queued root task starts and a terminal candidate snapshot
  at outcome derivation; blockers (pre-existing dirt, stale/missing
  baseline, failed scan) ride into review and acceptance evidence for the
  LLM panels to weigh — pre-existing owner work creates ambiguity a
  reviewing actor must see without the host inventing a semantic outcome. Do
  not turn blockers into structural outcome vetoes, and do not add a
  lease/holder service, a second ledger, or runtime writer keyword scanners.
- The acceptance packet reads mutation evidence from the canonical results
  root (`budget_drive_root` first), the same root the writer and the outcome
  consumer use.
- Git staging is attribution-based: `paths=None` means the clean-at-baseline
  candidate set, an explicit list must be its subset, and empty never means
  `git add -A`. Preserve pre-existing user dirt as excluded evidence.
  Whole-tree staging belongs only to typed managed update/release
  transactions and the typed external patch-capture transaction; contexts
  without a captured baseline keep the legacy staging contract.
- Resolve unversioned Python only for `run_command`, `run_script`,
  `start_service`, and run-kind `verify_and_record`, once BEFORE the shell
  guard; guard and handler receive identical argv. Resolve bare Node for the
  same four surfaces but once AFTER the dispatch gates — the node health
  check executes an argv[0]-steered candidate, and probing before the gates
  would run a planted PATH shim for a request the fences would refuse.
  Never rewrite explicit paths, versioned interpreters, shell bodies, or
  remote execution; never install a dependency in response to
  `ModuleNotFoundError`; with no usable runtime the argv runs as written and
  fails honestly (a rewritten absolute shebang is a disclosed residual).
- Skill Review ordinals and provenance stay in `review_job.json` and the
  append-only `review_history.jsonl`: allocate under the lifecycle lock,
  consume a round only after actual start, write one terminal row per
  `job_id`, and compute legacy ordinals at read time without rewriting
  history.

Enforcement: `tests/test_mutation_attribution.py`.

