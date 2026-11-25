# Delegated-run admission — threat model

**Schema floor is enforced at admission; the boundary is read back per attempt and
DISCLOSED, never required.** Owner: `ouroboros/config.py` (the two floors),
`ouroboros/subagent_route_health.route_health` (the admission decision),
`ouroboros/gateways/claudexor.attempt_containment` (the applied-fact reader) and
`ouroboros/tools/delegate_terminal_evidence.py` and
`ouroboros/delegate_start_instructions.py` (the three-place disclosure). This document is the reason
those numbers and that predicate are what they are; change it in the same commit as the code.

Claudexor owns the mirror document, `docs/DELEGATED_CONFINEMENT.md` in its own tree, which
describes the MECHANISM. This one describes what Ouroboros can know about that mechanism from
the outside, and what it says when the answer is "nothing was applied".

## 1. The asset

The token named by the selected daemon's `control-api.json` is a bearer for
the entire `/v2` control API. A process that reads it can request runs at other
access levels on registered projects. Child-scope instructions do not restrict
what a holder of the daemon token can ask the control API to do.

Ouroboros owns a daemon under `DATA_DIR/claudexor/`:
`CLAUDEXOR_CONFIG_DIR` relocates its complete config, credential, run and daemon
state. A separately configured daemon may use its own runtime home instead.
In either case, changing a child's `HOME` redirects home-relative lookups; it
does not make the actual token path inaccessible.

## 2. The actor

A delegated mutating child uses the shape from
`subagents.delegated_run_shape`: `mode: agent`, captured `access: full` or
`workspace_write`, `execution.isolation: live`, with the delegated marker. For Git and skill
payload work, `tools/delegate.py` provisions a private execution snapshot
before dispatch; a Git one is a linked `git worktree` at a commit git makes from
the target's current files (`subagent_worktrees.snapshot_commit`, shared by writer,
investigation and task copies, so shallow and partial clones work and a
target edit during provisioning cannot reach or refuse the copy). `live` means
that Claudexor works in that supplied execution root; it does not mean direct
edits to the authoritative target. The host captures the result as a commit,
can release the checkout after capture and named-evidence retention; native disposition
defers cleanup to custody GC. The existing integration path applies the result. The exception is the
starter's explicit `workspace: in_place` (Git writers only; owner 25.10): the
supplied root is the starter's own write root, edits land as the run makes them,
nothing is captured, applied or rejected, undo is through Git, the receipt and
`resource_ref` record the choice, and two in-place writers in one folder mix.
Ordinary folders use the separate supported direct/copy work-product shape in
`delegate_directory.py`.

`delegate_start(continue_from=...)` reuses an undisposed private snapshot with
the same root, baseline and binding. A released checkout is restored at handover
(captured files, HEAD at the baseline); only the successor's cumulative patch
is owed. For ordinary Git, a disposed patch means a fresh snapshot. For skill
payloads, omit the selector or assert the recorded bucket/name; a writer needs
the original snapshot and live baseline. Start fresh explicitly after disposition
or drift. The admission floors are unchanged:
omitted access keeps the run's narrower profile; explicit access never widens it.
A writable-to-readonly continuation replaces inherited execution with the ordinary
nondelegated envelope, no workspace override, and an empty auxiliary-roots list where
negotiated. Its current read scope stays; the old writer's undisposed patch stays separate.
The sole widening exception is an explicit
plan implementation by its recorded starter, within the original and still-current ceiling
(`delegate_plan.py`); missing legacy ceilings confer no permission. Plan answers/clarifications
stay read-only on their thread. Git/payload plans preserve a private copy for implementation;
plain-folder plans can implement in place under ordinary mutability admission. Selected
folder copies need a fresh executor: the plan thread cannot retarget its workspace.

The harness can execute model-generated commands under the operator's OS
identity. It is not assumed hostile, but the host cannot review each command
before it runs.

When a private snapshot exists, the host appends a separate typed execution
binding after the immutable inherited work order: the snapshot is the root of the
run's work product and the stable project root is a read-only identity until
explicit integration. Legacy engines exposing only the snapshot as `scope.root` get
the same binding. A resumed session may keep earlier instructions, so each access
line scopes run facts (access; edit, evidence, scratch, copy-local places; a copy's
read-only target) to one run and finds the current binding by identity, not position:
this turn's host facts, at the harness's starting directory; quoted ones are history.
Ordinary Continue restates access and binding (or directory). The binding, `workspace_write` access
line and boundary disclosure share the rule for verification output: disposable process state (test
scratch, package-manager temporaries, build or dependency output that only serves
checks) goes outside every Git working tree, the
binding naming safe_test's default parent through the launcher's own predicate,
so its ancestor guard stays intact (only a body copy's facts name safe_test itself);
output a tool must keep in the project tree stays only in new files Git ignores: generated
build/dependency output is not automatically excluded from the result-commit capture, Git
ignore rules keep new untracked files out, and tracking overrides them;
durable package downloads go only to the shared download cache the environment facts
name; evidence (reports, logs, sources, repro scripts, intended binaries of any size) goes
under the copy's `.review-evidence/`, whose retention applies no cache, binary, size or
ignore filter. Where the route's catalog row
negotiates `accessControl.auxiliaryWritableRoots` (never a guess from a harness name or
write mechanism) and the start is an Agent `workspace_write` one, the host makes the run's
short scratch at `<container>/w`, under `test_environment.session_scratch_parent()` (`/var/tmp/obs-<random>`
on macOS/Linux). The ungranted container's regular nonce marker and recorded address identify it;
whole-folder restoration there inherits its holder and cleanup. Its record is
`task_drives/<task>/session_scratch_owners/<snapshot>/scratch.json`, or a reviewer's
`session_scratch_owners/seat-<seat>/scratch.json` in its seat home. Legacy path/inode records remain unchanged.
Scratch and cache `uv`/`pip`/`npm` folders are real, outside Git, never reached through a link.
The host adds only these workspaces, never containers, markers or records, as
`execution.auxiliaryWritableRoots`; the binding, a seat's checking text and the facts then
name that scratch instead of the default parent (`workspace_preflight.session_service_directories`).
A `full` Agent start, whose rights already reach such a folder, gets the same owned scratch with nothing
sent: a private copy's (keyed by its snapshot), an in-place run's (`in-place-<invocation>`) and a reviewer seat's.
The binding declares scratch disposable before use (`DISPOSABLE_SCRATCH`). Create one-off measurement
snapshots there. Return the report, script, results, source and hash; explicitly return needed exact
bytes as evidence. Scratch follows its seat or task and parent decision through Continue.
Native profile rights are unchanged; the managed allocator makes only new container/marker metadata
service-readable, without widening the workspace grant. IPC or process inspection may still refuse;
keep its command and diagnostics. After inspecting the candidate, the integrating mind may run it through
existing host tools under its own authority. This runs candidate code with ordinary OS-user rights;
safe_test isolates environment and process custody, not the OS. No automatic Full, network change,
new verifier or weakened check follows from a refusal.
Any other route gets no such field and a typed limitation in its facts (at full access, a scratch
that cannot be made is `scratch_unavailable`); read-only, a `workspace_write` `tool_policy` route,
an in-place `workspace_write` run, an engine folder copy and a review thread's turns keep their
on-demand behavior, with no owned scratch; an exact
retry replays its recorded request, without recreating paths the OS or the original task's
drive may already have retired. A continuation binds its own start's rights; one from another
task need not reuse prior scratch; a new start replaces scratch removed by OS aging, never a
directory with a different identity. The drive's custody trash removal retires short scratch
too. Retention markers, unreadable/replaced ownership or failed removal keep the trash and
pointer; they may need manual resolution. A hard kill before record publication can leave an
unrecorded container with marker and workspace. No temp-root sweep or retry fallback is provided. A review
thread, whose turns carry geometry only, claims no grant.
Capture keeps files new there since the baseline, untracked, staged or committed, out of
the result commit, disclosed, and retains them whole beside it before the copy is
released. Baseline-owned entries, including symlinks, stay product and are neither
followed nor retained as host evidence; a link or special entry in the folder is
recorded and never followed. Only bytes a legacy budget left behind keep the
undisposed copy in custody as their last copy
(`subagent_integration_delegated.run_named_evidence`). A skill-payload
or engine directory product copy is captured whole and names no evidence folder; a
folder investigation's copy is itself its retained evidence. A writable Git copy or
reviewer seat also gets candidate-derived environment facts
(`workspace_preflight.session_environment_facts`), never an install; a read-only
session gets neither a write place nor those facts. Full native access does not make the
binding enforceable by itself; the locked apply check of the saved patch decides whether
integration is safe, and a result that holds exactly the files its copy started from
closes itself as `no_changes`. Excluded nested repositories remain outside the snapshot
inventory, are named on the start receipt and disclosed as an untracked residual
(an ignored clone is not inventoried, and a tracked submodule is not populated). What a
result cannot hold (a Git repository or a submodule the run checked out inside its copy, a
folder Git cannot open) is named, and the writer's checkout stays until apply or reject.

A child or reviewer without write authority but with configured
`workspace_write`/`full` access is an evidence-only investigation: the same
delegated `agent`/`live` shape in a private copy (the same linked Git worktree, or
the engine's directory copy), recorded with `authority_source: investigation`;
its capture is retained whole as evidence, never integrated, and the copy goes once
it is saved. The Git copy shares the source repository's refs, stash, hooks and
configuration (the former independent clone failed on shallow and partial
sources). A Git-subdirectory starting folder copies the enclosing worktree and
starts in the matching folder;
a folderless Project's engine copy (a second, engine-made one) is made from this invocation's staged
inputs (`delegate_readonly_inputs`, `resource_ref.source_root`), never from the
live scratch. Folderless readonly and investigation starts stage every permitted lineage file unless a nonempty
`input_files` selection replaces them (3b), a scheduled origin's whole `task_drive`/`artifact_store` included
(`tool_access_reads.scheduled_origin`), bytes only (`artifacts.copy_artifact_file`): an APFS or reflink clone
taken at one instant, else a bounded stream. An uncompressed record in `record_compression.in_scope` where compression is supported,
a store file and an immutable or append-only source always stream. Thus non-clone paths still write full copies. One
unreadable or nonregular file refuses the start before opening it (`readonly_inputs_unavailable`).
A folder no private copy serves (a data-drive subfolder, an
ignored starting folder) refuses typed, naming `access='readonly'`. A read-only child
(an explicit `readonly` lowering, Observe, or a Presence ceiling without a
delegated write grant) defaults to `mode: ask`, `access: readonly`. Explicit
`mode: plan` is also read-only; omitted mode preserves access-derived execution. Its parent-selected starting folder is the session project
root, not a read boundary; omitting it preserves inherited/folderless behavior. Read-only workers and
reviewer sessions are told so and read task-authorized sources within the parent's read authority
(`READONLY_SOURCE_SCOPE`); a frozen review stays on its snapshot.
The host reads effective access back for both shapes;
the delegated HOME/boundary checks below apply only to marker-carrying runs.

## 2a. Stable project identity and persistent registration

For Git and skill payload work, fresh mutating starts on an engine satisfying the workspace-root release
contract (`CLAUDEXOR_DELEGATED_WORKSPACE_ROOT_MIN_VERSION = "3.8.1"`) register and retain
the user's actual target project in `scope.root`, while the child's writable filesystem
rides separately as the private snapshot (for `in_place`, the target itself) in
`execution.workspaceRoot`. That registration
is the USER'S identity, not a disposable snapshot: it is marked `project_persistent`
(`delegate_registration_policy.persistent_registration` — stable execution workspace
plus `workspace_write`), and retirement honours the marker at settlement and proven-unsent
preparation. An error answering an entered POST preserves the registration: accepted-work
lookup may bind its run or job for ordinary observation and settlement; a failed or missing
lookup proves no non-admission. The ownership duty is discharged durably
(`PROJECT_RETIRED` with `project_kept: true`) without
deleting the project, and any persistent sharer makes the shared project undeletable for
its non-persistent siblings. Durable records written before the marker existed fall back
to the immutable stored request (`execution.workspaceRoot` + `access`) at recovery time
(`record_persistent`); rows that carry the key are authoritative and are never
recomputed from a live engine. Older engines keep the legacy snapshot-in-`scope.root`
shape: that host-minted snapshot declares `scope.ephemeral` and registers nothing where
the engine accepts the field (`CLAUDEXOR_EPHEMERAL_SCOPE_MIN_VERSION`), and otherwise
retires its one-shot registration as before. A target Ouroboros minted itself
(`delegate_registration_policy.minted_root`: a folder inside a task's own drive, a review
tree, a registry copy) is no user identity either and declares `scope.ephemeral` the same
way; a sticky thread on such a root does so from `CLAUDEXOR_EPHEMERAL_THREAD_MIN_VERSION`,
the engine release that sweeps the lanes those threads leave, and below it keeps the
registration the thread is bound to.

## 3. What Ouroboros actually controls

Ouroboros selects and delivers an immutable Claudexor runtime through
`claudexor_runtime_pin.json` and `claudexor_runtime.py`. Executable bytes live
under `DATA_DIR/state/cx`; credentials and daemon state remain separately under
`DATA_DIR/claudexor`. The reviewed pin selects the next spawn, while the serving
process may still run an earlier pin until its lifecycle ends. Admission uses
the engine version returned by the connected daemon's handshake.

Claudexor implements the harness boundary. Ouroboros controls admission,
execution-root preparation, custody and reporting through the control API; it
cannot infer an applied boundary merely from having delivered a particular
engine build.

The marginal escalation matters: this child already holds a shell in its
assigned worktree, running the operator's code as the operator. Access to the
daemon token adds control-plane authority, but withholding the whole lane
because a host has no boundary mechanism would also remove useful delegated
execution. The contract therefore checks required request support and reports
what each attempt actually received.

New configured sessions default to full native access; an explicit owner row or
invocation may lower it. Full requests no OS sandbox. The private execution
snapshot still owns patch delivery (an `in_place` run's edits are the delivery), and
explicit task constraints remain in force.
The owned gateway grants full access only for an absent scoped trust record,
preserving an existing denial. Older immutable snapshots without an access field
keep workspace_write; retries keep their exact recorded request. Review
sessions run at their seat's configured access in a private checking copy (a
catalog session row defaults to full), and explicitly read-only seats and tasks
retain readonly/ask. Scoped HOME and
actual-access receipts remain separate facts, and scoped trust grants persist
without an automatic cleanup policy.

## 3a. Selected task documents

Owner: `ouroboros/delegate_documents.py`, transport `ouroboros/gateways/claudexor_uploads.py`.
A task's retained inputs reach a delegated run as engine documents only when a start names
them: `attachment_paths` on `delegate_start`, or on an agent-session `schedule_subagent`
for the first start of that child's work order. Each entry is an exact manifest relpath
(`attachments/<name>`) of a staged, non-image row of the task's own inputs; images, rejected
or unknown rows refuse typed. Omission keeps the work order's input paths and readonly
copies and a byte-identical run body; `[]` adds none. API children refuse the selection. A
`declared` child receives only its selected inputs, disclosed in its work order.

A scheduled selection, its `attachment_import` and the new-import intent saved for `new`
are scheduling intent (`SUBAGENT_INTENT_FIELDS`): they cross admission, the queue snapshot
and the worker payload to the host's first start. When that start is refused unrun with an
unknown, in-flight or busy import, the child is woken with the refusal instead of ending
unrun, and its route stays available. Until a physical start exists (in that episode or a
restored one), a start without `attachment_paths` of its own is still the work order's first
start: it carries the saved selection, and the saved choice and intent unless it names its
own `attachment_import`. The saved intent begins only the first import of an input the task
holds no import record for; an input already imported, under that intent or by one of the
child's own starts (its explicit `new` included), keeps its current key, so a start that does
not ask for `new` itself never mints a second key for an input.

What crosses into the engine: the retained staged copy's bytes, read only for a fresh
intent or while the engine reports the upload `open` and verified against the manifest's
size and sha256, never the owner's original or any host path. The request names the file
by its manifest leaf and an opaque `ouroboros-task-input:<task>/<relpath>` source. Byte
acceptance is not parsing, OCR or understanding; the route's own `attachmentInputs`
(MIME, size, count) decide admission, and a missing keyed-import catalog descriptor
(`POST /v2/uploads/imports`), an unknown handshake `storeId` or a non-admitting route refuses
the start typed before any import or run. The canonical body then carries `attachments:
[{resourceId}]`; `retry_of` replays it and never reselects files.

Identity is one current host-minted random key per input, bound to the engine store it was
minted in: written with that store id, the input's manifest identity and the exact request
(including `expectedStoreId`) to the task's custody marks before any network call; a
possibly-sent mark precedes the first full declaration. Every later attempt, another run
included, replays lookup-only first, before any source read; the answer's `storeId`, bound
digest, kind, MIME and name must equal the recorded ones. An answer naming another store or
other bytes refuses as such; one too malformed to compare leaves the outcome unknown, so a
later start replays the same key. A default start never changes
store: an intent or inherited receipt bound to another store refuses typed
(`document_import_store_mismatch`) before any source read or import call, keeping its key.
A lookup 404 is a retained-binding gap and a cancelled, conflicting or content-rejected key
is finished: all refuse typed and never mint a key on their own. `attachment_import='new'`
is the explicit decision to begin another import in the serving store (another resource may
result); one new key per invocation, kept across a restarted host first start; earlier
records stay on the marks row and in the chain. Copied rows carry observed imports, gaps and
finished keys included (`engine_imports`, bound to the row's bytes, kind, MIME and name): children
keep the same identity and state, so only an explicit `new` can replace a finished import. Every schedule copying a non-image input reads the parent's
custody marks once, whether or not it selects anything, and refuses typed
(`document_import_custody_unavailable`) before the child exists when they cannot be read,
rather than letting a copy mint a fresh import; copies of images alone read nothing.
Scheduling still copies each inherited input's retained staged copy, so a gone copy refuses
the schedule (`SUBTASK_ATTACHMENT_ERROR`) even when its ready receipt could serve the child;
only a direct `delegate_start` replays a ready receipt without that copy. A `continue_from`
continuation inherits no documents and selects again; an owner Continue's successor copies
the work's inputs with its predecessor's observed receipts (unreadable marks refuse
`continuation_inputs_unavailable`), so its selection reuses them.
New stagings get the same Markdown and DOCX labels on every interpreter; rows recorded
earlier with an unprovable MIME (for example Markdown staged as `application/octet-stream`)
keep their label and refuse on routes that do not declare it.

Managed lifetime is the engine's opt-in, used only where its catalog lists the exact release
row (`POST /v2/resources/:id/release`, `ControlResourceReleaseRequest`); no version is
inferred, and without that row nothing sends `lifetime` or releases anything. There every
fresh intent declares `lifetime: "managed"`, bound into its key, so each replay (lookup-only
included) resends it; receipts keep it and name the task that owes the release
(`owner_task_id`), and earlier requests replay byte-unchanged. The intent's own REQUESTED
fact opens the obligation on the minting task's custody marks (`managed_imports`, by key,
never capped), so every explicit `new`, an unknown outcome and an import whose start never
ran stay owed until settled; a child, a successor or any other holder only replays the
receipt. The `runtime_retirement` pass reads the owner when its row is written terminal (or
its timeout retry's is), keeps it waiting while the release is held, and releases only when
nothing can name the resource in a new operation: the owner's tree has ended (no live,
unsettled or queued member, open review or pending delegated start), so has every root its
owner's Continue admitted after it with that root's tree, no Keep stands on the owner, and no
Continue is offered or still undecided. Unlike native registration retention, an author's
stop, a bound or unverifiable claim, an interruption whose cause is not recorded and an offer
whose budget close or deadline is unknown all hold; an ordinary completion is decided. An
obligation opened before that pass records a terminal-write floor consumed on its first pass;
an ended owner schedules the pass even without local files or START. Thread cutover keeps
the first pass's own time. After waits, closure is rechecked; release intent binds under
Keep's existing custody lock, HTTP runs outside it. Earlier Keep prevents release; later
Keep saves with `release_unconfirmed`. Managed imports remain Keep material without local
files. A key with no observed resource is looked up; an open upload is cancelled before its
obligation settles. Unknown cancellation keeps its identity for readback. Release names the
recorded store and digest; an unknown outcome, another serving store or an engine without the
row waits and is sent again unchanged on a later pass. A release closes future references
only: what an accepted command, turn, draft or export holds keeps its bytes. A `retry_of` that
re-sends the saved START key and body is that accepted operation, answered as such while the
engine keeps the command; a new operation naming the resource (a new start, `continue_from`,
the engine's Exact Retry, or that body once the command was pruned) refuses typed
(`resource_released`). An import replay of a released key answers `released`, finished like
`cancelled`, so only `attachment_import='new'` begins another import.

## 3b. Selected input files

`input_files` on `delegate_start` and read-only `schedule_subagent` (API or session)
accepts two entry shapes. Every entry is mandatory. A nonempty selection replaces
automatic folderless task/artifact lineage staging. Omission, null and `[]` keep
automatic inputs; only `[]` records an empty receipt. Prompt paths select nothing.
Engine documents use `attachment_paths` (§3a).

- `{path, root?}` selects a regular file under `active_workspace`, `system_repo`,
  `runtime_data`, `task_drive`, `artifact_store`, `user_files`, `subagent_projects` or
  `deliverables`. An absolute path infers its root as in `read_file`; the relative
  default is `active_workspace`. Other roots, including `skill_payload`, refuse.
  Packed checkpoint handles use normal read-back.
- `{"git_range":{"base":"HEAD^","head":"HEAD","path":"src"}}` selects the changes
  between two unambiguous local commits in the caller's repository. `root` defaults
  to `active_workspace` (also `system_repo`); `repository` selects a worktree within
  it, `path` a literal relative scope. Empty selectors use the whole root/worktree.
  Git peels once, retains a full-index binary patch with short gitlinks, without
  renames, textconv or external diff. Managed/Docker execution refuses: no equivalent
  path-policy binding is established for host comparison. `range.json` binds revisions,
  trees, paths (otherwise `path_base64`), status, and raw patch size/SHA-256; it names
  the lossless UTF-8 companion and inverse when text reads cannot preserve raw bytes.
  On Windows, an entry without an unambiguous native spelling refuses the whole selection.
  Retained bytes survive ref movement; a fresh Git rendering can differ.

`delegate_readonly_inputs` resolves each reader's own dispatch address; a shared
physical source does not carry the caller's logical grant into the receiver.
Both must admit read, copy and hash. `delegate_git_range` also checks fixed VCS
scopes and introspection of exact historical entries. A receiver-relative protected
directory still covers its historical descendants when absent from the worktree.
Only a scheduled child's own read of each exact copy address, a fixed file binding included, is
admitted before publication. Pre-queue work grows with
changed paths times the readers' inheritance depth. An entry resolving outside its repository, a
symlink under a Presence ceiling, or a path outside a reader's narrowed read policy refuses the range. Admission adds no
size or count limit beyond the capture timeout. One range reads one repository and stages into one session
directory; multi-root is not offered.

Every refusal precedes queueing or the request row and sends nothing. `delegate_start`
(`start_inputs`, each `definitely_unrun`) names four: `input_files_invalid`, a selection of
the wrong shape; `input_files_readonly_only`, a nonempty selection on a start that can write;
`input_files_unavailable`, with one row per selector (per product once a range is copied or verified), when an entry is denied, missing, not a
regular file, on an unsupported root, changed while copied or not copied; and
`readonly_inputs_unavailable`, when the host cannot stage (scratch, retained folder, a plan
thread that names no folder). Its check before the daemon answers the first and third for
ordinary files and defers Git comparison to retention. `schedule_subagent` reports the wrong
shape or an acting `write_surface` (`input_files_readonly_only`) as `TOOL_ARG_ERROR`, and an
entry or staging failure alike as `input_files_unavailable`. Each Git capture of a range is
bounded by a read timeout; a failed capture refuses its entry. An expired tool wait is none
of these: preparation may still finish as a late start result, so inspect that start before
requesting another.

The selecting task's artifact store retains verified bytes under
`source_handles/delegated_readonly_inputs/<invocation|scheduled-<child>>/inputs/selected/<sha256[:16]>/<name>`
with an `inputs.json` source map. A colliding occupied address refuses. Long paths
can still fail on Windows. Source handles survive drive retirement, stay out of
deliverables and automatic lineage walks, and can themselves be selected explicitly.
One receipt row names each retained file, its full hash, size, source and local path.
A range has one row per product. Prepared is not read. Native `read_file` reads
and hashes the entire file per page; smaller pages bound delivery, not file I/O.

A folderless session restages and verifies the files inside its scope (`carrier: scope`).
A bound folder keeps its scope/cwd and gets verified absolute paths (`outside_scope_path`);
the harness determines native read reach. A plan-thread turn keeps its recorded
`repoRoot`, stages no lineage, names new copies outside it and refuses a missing
folderless thread root. Continuations use only their own selection; their prompt
restates the map, even on an otherwise empty turn. A selection, unseen owner words
or promotion also restates typed access. `START_REQUESTED` and the started result
carry the same receipt. `retry_of` accepts no new selection and replays the recorded
request, paths and receipt.

`prepared_input_files` crosses admission, snapshot and worker payload; native
first input names each copy. The parent's store is lineage; under Presence,
`prepared_copy_read` admits a handed copy matching its digest. New orders mark
original selection with `work_order_input_files_original_only`; legacy orders
keep exact wording for recovery. Start instructions name effective copies. The
first start verifies the originals unless replaced (including `[]`); later starts
select their own. `resource_intent` keeps its contract: `explicit_resource` for a chosen workspace_root,
`explicit_none` for folderless. The nanny's rights differ from the leaf's.

## 4. Compatibility floors and applied evidence

The constants in `ouroboros/config.py` answer request-compatibility questions:

| Engine version | Request support used by Ouroboros | Consequence |
| --- | --- | --- |
| Below `CLAUDEXOR_MIN_VERSION` (3.2.0) | Below the supported control transport | Handshake refuses the route |
| From 3.2.0, below `CLAUDEXOR_DELEGATED_MARKER_MIN_VERSION` (3.3.0) | Read-only shape is supported; `execution.delegated` is not | Read-only delegation remains available; a mutating shape gets `engine_rejects_delegated_marker` |
| From 3.3.0 | Delegated marker is schema-compatible | Admission can proceed subject to route readiness; confinement is read from attempt evidence |
| From `CLAUDEXOR_DELEGATED_WORKSPACE_ROOT_MIN_VERSION` (3.8.1) | Separate `execution.workspaceRoot` is supported | Stable target registration and private execution root stay distinct (§2a) |

These floors are not a platform-support matrix. A version describes a build,
not what a particular attempt applied. Raising the marker floor to a release
that contains a boundary would still not prove that boundary exists on every
host; it would also refuse older engines that can execute with honest
unconfined disclosure. The report must instead follow the attempt evidence
in §8. The floors are compatibility minima, not a claim that the managed pin
or serving engine currently equals one of them.

## 5. Why a version at all, and why not a capability probe

The marker floor prevents a known request-schema failure before dispatch.
The capability catalog's top-level `runControlKeys` does not establish support
for the nested `execution.delegated` field. Its per-harness `delegation` object
describes MCP injection for Claudexor's own delegation strategy, which is a
different capability. `subagent_route_health.route_health` therefore does not
use that field as proof of marker support.

A behavioral test of the start endpoint would be the operation itself: sending
the field to an engine that accepts it starts a run. Admission uses the
compatibility constant instead of spending a model run to probe that schema.

For the BOUNDARY question no probe is needed, because the engine already answers it — after
the fact, on the attempt record (§8). That answer is a fact about the run rather than a
prediction about a build, which is why it, and not the version, is what the report is built on.

## 6. The rule

Two floors, because they gate two different lanes, and one evidence reader, because there is
one question left that a floor cannot answer:

- `CLAUDEXOR_MIN_VERSION` (3.2.0) — the TRANSPORT floor, checked at handshake. It gates
  read-only delegation and must be the lowest engine that serves it. Fresh read-only starts
  omit `execution`; lowering a writable continuation explicitly replaces it (§2).
- `CLAUDEXOR_DELEGATED_MARKER_MIN_VERSION` (3.3.0) — the MARKER floor, checked in
  `route_health` against the run SHAPE, before a token is spent. An engine below it would
  reject the request with a 400, so the lane refuses it with a typed reason
  (`engine_rejects_delegated_marker`) instead of spending a dispatch on a certain failure.
- `attempt_containment` — the applied-evidence reader. Its boundary evidence
  feeds disclosure, never a boundary-required admission gate. Its HOME facts
  also feed the separate breach check in §8.

An engine between the two floors serves read-only delegation and refuses mutating delegation.
Keeping the marker floor separate preserves that serving read-only lane.

Both floors fail CLOSED: `engine_at_least` compares an absent or unparsable version as `(0,)`,
below every floor.

Refusal never degrades into metered native execution on a PIN. An explicit `executor="harness"`
request that cannot be served becomes `blocked` — `agent.executor_blocked_outcome` ends the
child unrun — because silently spending API money is the one outcome an explicit pin must
never produce. A child without a configured subagent row runs natively with no marker; `auto` is not a request
(owner decision D23).

## 7. What this does NOT cover

Stated plainly, because a floor described as total is worse than a narrow one.

- **Not the enforcement.** Ouroboros admits; the engine confines. The floor is a claim about a
  build, checked against a self-reported number, and it is used only for the schema
  question, where that is enough.
- **Not a lying or downgraded daemon.** The version is self-reported over loopback, and so are
  the applied facts on the attempt record. Anything that can forge either already runs as the
  operator and has the token.
- **Not the engine implementation.** Claudexor is built separately and selected
  by an exact reviewed runtime pin. Its release must actually implement the
  request shape its version promises. The compatibility floor checks that
  declared contract; it does not inspect the engine's code at dispatch.
- **Not a promise that anything is confined.** A delegated mutating run is allowed on a host
  with no boundary mechanism at all. What is guaranteed is that the run is not DESCRIBED as
  confined when it is not — the disclosure, not the boundary, is the invariant.
- **Not what the boundary itself leaves open where it does exist.** The vendor credential root
  stays readable to the child and the network is not fenced. Those are the engine's to state
  and it states them in `docs/DELEGATED_CONFINEMENT.md`. Ouroboros must not re-describe
  them as covered.
- **Not the read-only lane's confinement.** A read-only child is scoped by Claudexor's ordinary
  envelope. Ouroboros asks for no marker and verifies no boundary there.
- **Not an engine that applied a boundary and recorded nothing.** Silence is read as "no
  boundary", so such a run is disclosed as unconfined when it was in fact confined. That is
  the honest limit of an applied-fact reader, and it is the safe direction: the consequence is
  a disclosure, never a refusal.

## 8. Evidence, not intention — and the disclosure it feeds

What the run actually got is read back from the run's own artifacts
(`<runDir>/attempts/<id>/attempt.yaml`). The HOME pair is artifact-only — the engine projects it
onto no `/v2` response — while the boundary is also on the run detail, as
`candidates[].confinement` (`proven` / `mechanism` / `verifiedDeniedPath` / `unavailableReason`); the artifact stays the one reader here because it answers both halves at once.
Two facts, one reader (`gateways.claudexor.attempt_containment`):

- the HOME pair, `harness_home_isolated` / `harness_home_dir`;
- the boundary, `confinement_mechanism` together with `confinement_verified_denied_path` —
  the path the policy was executed against, and refused, on this host, for this attempt,
  before the harness ran.

**A mechanism without its proven path is not evidence.** The pair is read together and a
mechanism named alone reads as no boundary at all, because "confined: true" with nothing behind
it is exactly the promise the applied-fact block exists to replace.

**The mechanism is an opaque string.** Ouroboros keeps no list of mechanism names and no OS
test: the predicate is "did this attempt report a boundary it can prove", never "which platform
am I on". A boundary shipped for a second OS is therefore already handled, and a platform
branch would have gone on reporting "no boundary" forever after that day.

**The two halves take different rules about silence, on purpose.** A missing HOME fact stays
UNPROVEN rather than false, because the consequence of "false" there is a CANCELLATION, and an
attempt can legitimately record no `harness_home_isolated` — it is the one optional member of
the applied facts, omitted when the attempt died before its home was decided (and an older engine may omit
those facts from `attemptFailureRecord`). A missing mechanism
collapses to "no boundary", because the consequence there is a DISCLOSURE. Each silence is read
in the direction whose failure mode is recoverable.

**Confirmed HOME failures are distinct from missing evidence.** For attempts
that record the HOME isolation flag, `_home_isolation_breach` reports a breach
when that flag is false, or when the claimed isolated home resolves to the
operator's own home. A missing flag is skipped by this enforcement check and
remains unproven in the report. A scoped home nested under the operator's home
is not a breach, with or without an OS boundary: nesting is the engine's
ordinary layout and its absence of a boundary is disclosed rather than used
to cancel useful work. The engine's `confinement_unavailable_reason` amplifies
that disclosure; it never excuses a recorded false.

The run-level report also preserves partial evidence. `verified` remains false
unless every recorded attempt discloses its HOME fact, no HOME breach exists,
the HOME is not nested under the operator's, and all attempts name the same
proven boundary mechanism. `nested_under_operator_home` stays visible even if
a boundary was applied: the boundary is evidence of confinement; the HOME
redirect alone is not.

The disclosure reaches three places:

1. **the durable record** — a `delegate_run_unconfined` event when no boundary
   is reported or the HOME is nested under the operator's, once per run, carrying the
   note the parent was given, so the forensic trail of an integrated patch says where the
   work came from;
2. **the child's own prompt** — its instructions state that the boundary is a REQUEST and not
   a fact, that it must work as though there is none, and that it must not describe itself as
   sandboxed. It cannot be told which way it went, because nothing at start knows: the engine
   decides per attempt and records the fact afterwards;
3. **the parent-facing result** — `delegate_wait`'s terminal payload carries `containment`
   with `os_boundary`, `verified`, and `disclosed`/`attempts`, and a note naming what was
   reachable rather than merely saying a check failed.

## Independent runs under one focus

A root or configured-session nanny may own any number of delegated runs, on the
same configured row or different rows. Admission has no concurrency cap or
quota-derived team limit. Each deliberate prompt gets a fresh immutable invocation;
`retry_of` reuses the unresolved invocation's original body and binding. Existing
runs and undisposed patches remain custody obligations, not start refusals.

The initial leaf-first bootstrap alone uses the saved assignment and row. Its
identity is persisted under the actor decision lock before transport, so restarting
cannot bootstrap a second leaf. Omitted selectors retain the assigned-row behavior;
an explicit later selector and prompt describe independent work. Empty-prompt
replacement requires verified settlement. Zero-run race checks and unreadable
custody/billing refusals remain. A continuation inside the same snapshot still
requires an unambiguous predecessor disposition and a single writer; in-place
writers have no such floor, and two in one folder mix their edits.

Supervision schema 2 stores each run's cursor, pending wake and interaction ACKs
separately; mail and coordination cursors belong to the task. `delegate_wait` takes
one `run_id` or an exclusive `runs` array and wakes on any selected run. Exact
receipt incorporation alone ACKs a wake. An unknown-provider nanny hold may name
several positively live leaves, without resending the lost cognitive request.

Recovery schema 2 holds all run and pending-invocation bindings. Adoption reports
adopted, deferred or refused per holder, restoring a settled wake alongside live
and pending work. Saved cognition defers pending replay; other recovery first reads, never
creating, the saved binding's accepted work (`delegate_accepted.py`; live and paused owners too;
admitted apart from replay, a process Stop halting both) and binds it once: no STARTED is published
outside the actor decision lock, and a run only its process holds records money only on that row.
Automatic replay needs a plain miss and admitting-store receipt; explicit retry keeps its store rule.
Recovered custody carries the original request time and cap separately from STARTED publication:
review continuation and copy retention use that window, so late collection grants no fresh time.
Root-direct recovery uses the same path.
Orphan protection and registration retention name exact handles, not every run of a
reserved task. Stop reaches all owned runs and late pending starts; no new POST.
Windows transfer needs observed parent exit 42 and measured parent/successor identity.
Schema-1 checkpoints migrate in memory, writing v2 on mutation without rewriting events
or issuing requests. Corruption stays unknown; live v2 custody prevents downgrade.

Private skill-payload captures may coexist. The target mutation lock spans the
whole-payload content-hash check, apply and publication; a stale second capture
refuses with evidence retained. Git integration retains its target lock and
per-path baselines: disjoint changes can apply, overlapping drift conflicts.
Each starter still disposes each capture. There is no automatic merge arbiter.
These rules apply to API-only, subscription-only and mixed installations, outside
Cyber Pro and in it; configured route availability and review floors
remain independent.
