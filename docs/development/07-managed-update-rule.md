# Managed Update Rule

This chapter owns what a managed update may do to the live tree: how one exact official target is chosen and bound, which windows refuse conversation, how dirty local work is stashed rather than merged into history, what the authorized resolver may stage, and when a fresh rescue must precede destruction. It exists because a managed update is the one path that rewrites the live tree in place, so every step of it needs a stated custody rule.

- Keep the local work branch and the official update feed separate; the
  channel and branch topology live in ARCHITECTURE "8. Git Branching, CI,
  and Build" (`ouroboros/update_channels.py`).
- A preflight chooses one exact official target SHA. Apply binds to the
  disclosed base/target, closes new writers, drains direct turns,
  stops workers and tracked services, then re-plans before mutation. Write
  the update transaction before mutation; reopen writers only after a
  verified abort/rollback or a healthy restart. Delayed evolution cleanup
  acquires the same update lock and honors the same admission owner.
- Conversation admission is separate from repo-writing permission. The chat
  lanes refuse turns only during destructive windows (apply/replace/rollback
  prologue, materialization). While the ONE authorized assisted resolver holds
  the repository (`assisted_resolution` / `committing_assisted`) Main keeps
  answering, the registry guard still refuses repo tools to every other task,
  and steering reaches the resolver through the ordinary `steer_task` mailbox
  (`supervisor/worker_chat_lane.py::conversation_admitted_during_update`).
  The server's first-party packages are discovered and imported BEFORE conflict
  markers land in the live tree (`preload_owner_control_path`, called after the
  resolver readiness proof and before a boot re-materialization). Discovery uses
  the imported package paths; the packaged server runs embedded Python from the
  materialized Git repository. Tool admission remains owned by the registry.
- Dirty local work never enters merge history: the apply stashes it and
  restores it as uncommitted content; a conflicting restore keeps the stash
  and discloses the recovery command. The reviewed assisted resolver runs
  only when Git reports a real conflict; filenames do not create a second
  update policy. Managed materialization and rollback run their internal
  `git reset --hard` without interactive confirmation; the
  explicit-confirmation rule applies to the owner-facing generic restore
  seam.
- The authorized resolver stages the complete merge including tracked binary
  files, and review receives their exact staged mode/blob/size plus the
  parent object ids; missing exact metadata blocks. This exception does not
  weaken the ordinary commit pipeline's binary policy.
- A managed merge commits only with proof that the full suite ran green on
  the exact candidate tree ("The commit gate mirrors the CI split" names the
  proof authority). Any non-commit terminal of the resolver rolls the live
  tree back and best-effort preserves the attempt on the deterministic
  failed-update branch; the fresh rescue snapshot, not that branch, is the
  carrier rollback itself depends on.
- Take a fresh rescue before every destructive rollback and before boot-resume
  re-materialization, fail-open but disclosed durably at capture time, with its
  pointer recorded in the update transaction (the choke point and why the
  pre-update snapshot is not enough:
  `docs/architecture/02-startup-onboarding-flow.md`).
- Manual Restore reuses the same writer fence and pins the previous HEAD on
  a local recovery branch before reset. Promotion resolves the development
  SHA once and uses that exact SHA for both the local QA ref and any remote
  push.

Enforcement: `tests/test_update_merge_policy.py` (what the merge policy
refuses), `tests/test_update_dirty_stash.py` (the dirty-tree path),
`tests/test_update_hardening.py` and
`tests/test_update_tx_corrupt_quarantine.py` (transaction integrity and
quarantine).

