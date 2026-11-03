# Managed Update Rule

Managed updates rewrite the live tree under owner authority. Persist target, writer/local-work custody and resolver/recovery authority before mutation for replay.

- Separate the local work branch from the official feed (`ouroboros/update_channels.py`; ARCHITECTURE §8).
- Preflight reads saved prose/qualification for its exact pins (missing checks and accepted FAILs disclosed, no veto: `tests/test_update_qualification.py`); apply binds base and target. Smart, replace and Restore check the effective index lock before work stops; lock/read failures return not-applied, untouched. Apply closes writers, drains direct turns, stops workers/services, re-plans; reopen after verified abort/rollback or healthy restart. Evolution cleanup shares lock/admission.
- Destructive apply, replace, rollback and materialization refuse turns; in `assisted_resolution`/`committing_assisted`, Main answers but other tasks' repo tools stay refused (`worker_chat_lane.conversation_admitted_during_update`); steer through `tell_task`.
- Stash dirty work outside merge history, then restore. Never automatically drop an update stash, even on success: `rescue-local-*` pins it for Git cleanup. Only Git conflicts invoke the reviewed resolver; filenames create no policy.
- Persist restoration intent before apply; consume it on completion (restored, preserved or incomplete). Failed persistence keeps the marker; interrupted replay never resets/reapplies over returned or later work (`supervisor/update_merge.py`).
- Schema 1 is default; stash restoration and rollback with stashed work need schema 2 before checkout: older strict readers refuse interrupted new state without resetting. `stash_restored` records successful apply, never intent. Restart the desktop app, Docker/service or direct server, not its tab; in-app Restart refuses pending restoration.
- The resolver stages the merge, binaries included; review needs exact staged mode/blob/size and parent ids. Missing metadata blocks.
- Managed merge commits need a green full suite on the exact candidate ("The tiered commit and CI gates"). Non-commit resolver terminals roll back and best-effort preserve the failed-update branch; recovery requires fresh rescue. Boot holds Paused commits (`postcommit_resume`); Stop rolls back.
- Fresh rescue precedes destructive rollback and boot re-materialization; capture failure is fail-open, durably disclosed with a transaction pointer (ARCHITECTURE §2).
- `git_observation.host_git_env`: content reads never lock/refresh/repair the index. Writers keep mandatory locks/repair. Refusals keep stderr; no tree effect means not-applied.
- Manual Restore fences writers, pins HEAD, never pushes, interrupts without restart handoffs. Promotion binds one development SHA for QA and push. After-update facts reach panel and mind; the host never messages the owner (`tests/test_update_generation_facts.py`).
- Apply shows the engine transition, active work (unreadable: unknown), installed/bundled Python/SQLite, forward-only engine data; `engine_pin_policy` also drives restart.
- `git_bootstrap.json` needs identical clean checkout/dependency fingerprints; checkout/rollback invalidates it, successful smoke recreates it (ARCHITECTURE §2). Test zero-reset/zero-smoke repeats and change/transaction invalidation.

Enforcement: `tests/test_update_merge_policy.py`, `tests/test_update_dirty_stash.py`, `tests/test_update_stash_recovery.py` (real Git, crash/replay, Restart), `tests/test_update_hardening.py`, `tests/test_update_tx_corrupt_quarantine.py`.

Native hosts add no source updater: launcher hooks verify the artifact before core start and read back adoption before finalizing boot. Failure rolls back; worker self-restart cannot complete native adoption even when core SHA matches.
