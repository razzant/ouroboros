/** Dependency-free JSDoc mirror of the Dashboard storage envelopes homed in `ouroboros.storage` and
 *  re-exported by `ouroboros.gateway.contracts` (`GET /api/storage`, `POST /api/storage/measure`,
 *  `POST /api/storage/measure/cancel`, `POST /api/storage/free/preview`, `POST /api/storage/free`,
 *  `POST /api/storage/free/cancel`, `GET /api/storage/free/items`; `view=manual` lists the finished copies discarded by hand). */

/**
 * One depth-1 entry of a place's roots; files directly in a root form the part `(files)`.
 * @typedef {Object} StoragePart
 * @property {string} name
 * @property {number} bytes
 * @property {number} files
 */

/**
 * The parts beyond the largest eight, summed; the engine's `projects` and `profiles` are listed whatever their rank.
 * @typedef {Object} StorageRest
 * @property {number} count
 * @property {number} bytes
 */

/**
 * One counted place. `bytes` is null (unknown, never zero) when none of its present roots could be listed;
 * `growth_bytes` is against the previous finished count, present only when both counts hold the place.
 * @typedef {Object} StoragePlace
 * @property {string} id  // model_calls | engine | task_results | archives | memory_skills | project_folders | deliverables | work_folders | working_copies | app_state | data_other | program
 * @property {string} group  // records | engine | work | temporary | other
 * @property {string[]} roots
 * @property {boolean} present
 * @property {boolean} same_volume  // on the data folder's disk
 * @property {number|null} bytes
 * @property {number} logical_bytes
 * @property {number} files
 * @property {number} unreadable  // entries or folders that could not be read; the walk went on
 * @property {StoragePart[]} parts
 * @property {StorageRest} rest
 * @property {(number|null)=} growth_bytes
 * @property {'unreadable'=} growth_flag  // either count had unreadable items here
 * @property {string=} undecided  // why its default folder cannot be chosen: unknown, never empty
 */

/**
 * Free and total space of the data folder's volume, never a machine total.
 * @typedef {Object} StorageVolume
 * @property {number|null} free_bytes
 * @property {number|null} total_bytes
 * @property {string} observed_at
 */

/**
 * The last finished count with growth against the one before it.
 * @typedef {Object} StorageCount
 * @property {string} started_at
 * @property {string} finished_at
 * @property {number} duration_seconds
 * @property {string} io_priority  // the disk priority the count child ran at
 * @property {'allocated'|'logical'} basis  // what `bytes` counts
 * @property {number} bytes
 * @property {number} logical_bytes
 * @property {number} files
 * @property {number} unreadable
 * @property {StorageVolume} volume  // as observed when the count finished
 * @property {StoragePlace[]} places
 * @property {(number|null)=} growth_bytes
 * @property {string=} growth_since
 * @property {'unreadable'=} growth_flag
 */

/**
 * A running count: where it is and the places already finished in this run.
 * @typedef {Object} StorageCounting
 * @property {string} started_at
 * @property {string} place
 * @property {string} current
 * @property {number} places_done
 * @property {number} places_total
 * @property {number} bytes
 * @property {number} files
 * @property {StoragePart[]} done  // known-size finished places; `name` is the place's id
 */

/**
 * @typedef {Object} StorageError
 * @property {string} at
 * @property {string} message
 * @property {boolean} stopped  // the owner stopped it, as opposed to a failure
 */

/**
 * One of the largest items a free-space group offers.
 * @typedef {Object} StorageFreeItem
 * @property {string} id
 * @property {string} name  // the task's title when it has one, else the folder name
 * @property {number|null} bytes
 */

/**
 * One group the free action offers: finished tasks' work folders, working copies of finished work, and
 * leftovers no background pass removes (reviewers' files, ownerless review copies, unrecorded copies,
 * leftover environments and test-run records, Git marks of results never applied).
 * @typedef {Object} StorageFreeGroup
 * @property {string} id  // work_folders | working_copies | review_evidence | review_checkouts | lost_copies | service_folders | result_pins
 * @property {number} count
 * @property {number} inside_wait  // finished less than retention_days ago
 * @property {number} in_use  // live, pending, unknown or non-terminal: offered nowhere, named here
 * @property {number} unreadable  // no readable task record or custody: shown, never freed
 * @property {number|null} bytes
 * @property {boolean} bytes_complete
 * @property {StorageFreeItem[]} items
 */

/**
 * A typed reason and how many items it holds, for what a pool leaves out and what a run kept.
 * @typedef {Object} StorageFreeKept
 * @property {string} reason  // results_pending | in_use | busy | unreadable_record | restart | capture_unproven | unproven_excluded_inputs | retained_evidence | undeclared_environment | owner_keep | changed | other; and, only for what a pool leaves out, advanced_group (offered as a whole saved-work group)
 * @property {number} count
 * @property {number=} manual_count  // a pool's row only, when above zero: how many of `count` can be discarded by hand (view=manual). Never subtract the pool's global manual count from a row: each row says its own.
 */

/**
 * The ordinary pool of one plan: `null` at idle, may be present during `listing` with `bytes` still null or partial, final at `ready`.
 * `environment` is a whole temporary working environment, never installed packages alone.
 * @typedef {Object} StorageFreePool
 * @property {number} count
 * @property {number|null} bytes  // observed size, never a promise of space freed; null = unknown, never zero
 * @property {boolean} bytes_complete
 * @property {{work_folder?: number, working_copy?: number, environment?: number}} kinds
 * @property {StorageFreeKept[]} excluded  // same reason vocabulary as a run's kept items
 * @property {{count: number, bytes: number|null, bytes_complete: boolean}} manual  // finished copies the owner may discard one by one, never in `count`; the door shows when `count` > 0; bytes: null = unknown, never zero
 */

/**
 * What discarding one finished copy by hand would lose, `view=manual` only. "Not proven saved" means no verified capture holds
 * those paths; it does NOT mean they are unique or the only copy. The whole folder goes, never selected paths.
 * @typedef {Object} StorageFreeManual
 * @property {number} not_proven_saved_count  // all such paths, including those not listed; 0 only when the copy's sole unique work is commits on a branch that stays
 * @property {number} not_proven_saved_bytes  // sum of lstat sizes, always whole (a path that cannot be read withholds the whole offer)
 * @property {{path: string, bytes: number}[]} not_proven_saved  // the largest first, at most 50, paths relative to the copy
 * @property {number} omitted_count  // not_proven_saved_count minus the listed ones: say so when above zero
 * @property {boolean} keeps_commits  // its branch or pin stays, with any commits it holds
 */

/**
 * One filesystem object of a plan. `id` is opaque and valid inside this plan only.
 * @typedef {Object} StorageFreeEntry
 * @property {string} id
 * @property {string} group  // the id of its StorageFreeGroup
 * @property {'work_folder'|'working_copy'|'environment'|'other'} kind
 * @property {string} name
 * @property {string} path
 * @property {string|null} task_id
 * @property {number|null} bytes  // observed size (the whole folder); null = unknown, never zero
 * @property {boolean} bytes_complete
 * @property {string|null} eligible_after  // the earliest the background may consider it, not a promised deletion; null = never
 * @property {string|null} reason  // view=excluded and kept outcomes
 * @property {string|null} outcome  // view=outcomes only: failed | stuck | kept | not_started | deleting | removed
 * @property {'ordinary'|'saved_group'|'manual_discard'|'none'} availability  // how it can go: the pool, only a saved group, by hand (listed in view=excluded with its reason AND in view=manual), or not now. On view=outcomes it describes the preview, not the result: read `outcome`
 * @property {StorageFreeManual=} manual  // view=manual only
 */

/**
 * `GET /api/storage/free/items?plan=&view=&cursor=&limit=&version=`: one page. The pool and the excluded list are
 * immutable once ready (their version is the plan id); outcomes settle and can reorder, so a moved version answers
 * 409 `view_changed` and the client restarts from `cursor=0`. Other 409s: `preview_outdated`, `listing`; 400: `bad_view`.
 * @typedef {Object} StorageFreeItemsResponse
 * @property {string} plan
 * @property {'pool'|'excluded'|'manual'|'outcomes'} view
 * @property {string} version
 * @property {number} total
 * @property {Object<string, number>} counts  // by reason or outcome
 * @property {number|null} next_cursor
 * @property {StorageFreeEntry[]} items
 */

/**
 * The result of one run as it stands now: a moved folder is removed once its trash entry is gone.
 * @typedef {Object} StorageFreeRun
 * @property {string[]} groups
 * @property {number} selected
 * @property {number} done
 * @property {number} removed
 * @property {number} removed_bytes
 * @property {number} deleting  // moved aside, the maintenance child has not deleted them yet
 * @property {number} deleting_bytes
 * @property {number} stuck  // moved aside and survived the child's removal; retried at the child's next run
 * @property {StorageFreeKept[]} kept
 * @property {number|null} free_before
 * @property {number|null} free_after
 * @property {boolean} stopped
 * @property {string|null} error
 * @property {number=} failed  // items whose deletion failed
 * @property {number=} not_started  // unreached after a stop: untouched
 * @property {boolean=} outcomes_ready  // `view=outcomes` answers for this run
 */

/**
 * The free-space tray: a pure preview under a plan id, then a run of exactly the previewed items.
 * @typedef {Object} StorageFree
 * @property {'idle'|'listing'|'ready'|'running'|'done'|'failed'|'stopped'} state
 * @property {string|null} plan
 * @property {string|null} started_at
 * @property {string|null} finished_at
 * @property {number} done  // folders sized so far (listing) or attempted (running)
 * @property {number} total
 * @property {StorageFreeGroup[]} groups
 * @property {(StorageFreePool|null)=} pool
 * @property {StorageFreeRun|null} run
 * @property {string|null} error
 */

/**
 * `POST /api/storage/free`: the plan id the preview answered with and exactly one form: the whole groups to free of it,
 * the exact items of the ordinary pool, or the exact copies of `view=manual` with `accept_loss: true`; two forms together
 * answer `selection_conflict`. There is no "select all" on the wire: the client sends each id.
 * @typedef {Object} StorageFreeRequest
 * @property {string} plan
 * @property {string[]=} groups  // ids of StorageFreeGroup
 * @property {string[]=} items  // ids of StorageFreeEntry; any bad id refuses the whole request
 * @property {string[]=} manual  // ids of view=manual entries, each one the owner chose; not offered, unknown or outdated refuses the whole request
 * @property {boolean=} accept_loss  // exactly `true`, only after the owner was told the whole copy goes with whatever is not proven saved in it; anything else answers `loss_not_accepted`
 */

/**
 * One remover of a place, read fresh: a maintenance job's row or the engine's last retention pass.
 * @typedef {Object} StorageRemover
 * @property {string} id  // engine | runtime_retirement | custody_trash | evidence_packing | record_compression
 * @property {string} state  // a job: pending | running | done | failed | never | unsupported | unavailable; the engine: completed | held | truncated | failed | none | unavailable | unsupported
 * @property {string|null} at  // the job row's last write; the engine pass's end
 * @property {{class: string, at: string}=} last_error  // a job whose last run threw, until a run returns normally
 * @property {number=} compressed  // record_compression: files compressed so far in this place's plane
 * @property {(number|null)=} saved_bytes  // record_compression: that plane's allocation saved; null where unreported
 * @property {number=} freed_bytes  // the engine pass
 * @property {number=} held  // the engine pass: folders it held (project roots and native homes)
 * @property {string=} held_since  // the earliest of those holds
 * @property {number=} errors  // the engine pass's errors
 * @property {(string|null)=} first_error
 */

/**
 * The passive projection; the measure and free POSTs answer it too, with `started` (and the free run's
 * `reason`). `count` is omitted when `?count=<finished_at>` named the latest finished count.
 * @typedef {Object} StorageResponse
 * @property {(StorageCount|null)=} count
 * @property {StorageCounting|null} counting
 * @property {StorageError|null} error  // the last count's end; the saved count stays
 * @property {StorageVolume} volume  // fresh at this read
 * @property {number} retention_days
 * @property {StorageFree} free
 * @property {Object<string, StorageRemover[]>} removers  // place id -> its removers; a place nothing removes is absent
 * @property {boolean=} started
 * @property {string=} reason  // why a free run did not start: preview_outdated | busy | unknown_group | unknown_item | not_offered | nothing_selected | selection_conflict | loss_not_accepted
 */
