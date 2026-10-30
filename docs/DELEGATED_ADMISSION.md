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
releases the checkout once that capture and its named evidence are saved, and
applies the result through its existing integration path. The exception is the
starter's explicit `workspace: in_place` (Git writers only; owner 25.10): the
supplied root is the starter's own write root, edits land as the run makes them,
nothing is captured, applied or rejected, undo is through Git, the receipt and
`resource_ref` record the choice, and two in-place writers in one folder mix.
Ordinary folders use the separate supported direct/copy work-product shape in
`delegate_directory.py`.

One start provisions nothing: a continuation (`continue_from` on `delegate_start`,
`delegate_continuation.py`) of a run whose private snapshot is still undisposed
runs in THAT snapshot, with the same execution root, baseline and binding (a
checkout released after its capture is restored at the hand-over: the captured
files with HEAD at the baseline), and
takes its custody over, so the earlier work is captured once in the successor's
cumulative patch. A predecessor whose patch was already applied or rejected is
continued from a fresh snapshot of the target. The admission floors are unchanged:
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
Each ordinary continuation's prompt restates its access and binding (else its directory). The binding, the `workspace_write`
access line and the unproven-boundary sentence state one write rule, because
verification needs writes no copy may hold: disposable process state (test
scratch, package-manager temporaries) goes outside every Git working tree, the
binding naming safe_test's default parent through the launcher's own predicate,
so its ancestor guard stays intact (only a body copy's facts name safe_test itself);
durable package downloads go only to the shared download cache the environment facts
name; evidence goes under the copy's `.review-evidence/`. Where the route's catalog row
negotiates `accessControl.auxiliaryWritableRoots` (never a guess from a harness name or
write mechanism) and the start is an Agent `workspace_write` one, the host makes the run's
exact short scratch under `test_environment.default_temp_parent()` (`/tmp/obs-<random>` on
macOS, resolved physically), with its path/inode record in the ungranted task drive at
`task_drives/<task>/session_scratch_owners/<snapshot>/scratch.json` (a reviewer seat's
`seat-<seat>`), and the cache's `uv`/`pip`/`npm` folders, real directories outside every Git
tree and never reached through a link, and adds only these exact directories as
`execution.auxiliaryWritableRoots`; the binding, a seat's checking text and the facts then
name that scratch instead of the default parent (`workspace_preflight.session_service_directories`).
The native profile's existing temporary-directory permissions are unchanged. Directory grants do
not grant IPC or process inspection: a blocked check stays unverified, with its command and
diagnostics retained. After inspecting the candidate, the integrating mind may run it through
existing host tools under its own authority. This runs candidate code with ordinary OS-user rights;
safe_test isolates environment and process custody, not the OS. No automatic Full, network change,
new verifier or weakened check follows from a refusal.
Any other route gets no such field and a typed limitation in its facts; read-only, full
access and a `tool_policy` route (an unconfined shell) keep their on-demand behavior; an exact
retry replays its recorded request, without recreating paths the OS or the original task's
drive may already have retired. A continuation binds its own start's rights; one from another
task need not reuse prior scratch; a new start replaces scratch removed by OS aging, never a
directory with a different identity. The drive's custody trash removal retires short scratch
too. Retention markers, unreadable/replaced ownership or failed removal keep the trash and
pointer; they may require manual resolution, not just another pass. A hard kill before record publication may leave an
unrecorded empty temp directory. No temp-root sweep or retry-time fallback is provided. A review
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
a folderless Project's engine copy is made from this invocation's staged lineage
inputs (`delegate_readonly_inputs`, `resource_ref.source_root`), never from the
live scratch. Folderless readonly and investigation starts stage every permitted
lineage file, a scheduled origin's whole `task_drive`/`artifact_store` included
(`tool_access_reads.scheduled_origin`): the copy grows with that scratch, and one
unreadable or nonregular file refuses the start before opening it (`readonly_inputs_unavailable`). A folder no private copy serves (a data-drive subfolder, an
ignored starting folder) refuses typed, naming `access='readonly'`. A read-only child
(an explicit `readonly` lowering, Observe, or a Presence ceiling without a
delegated write grant) defaults to `mode: ask`, `access: readonly`. Explicit
`mode: plan` is also read-only; omitted mode preserves access-derived execution. Its parent-selected starting folder is the session project
root, not a read boundary; omitting it preserves inherited/folderless behavior.
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
plus `workspace_write`), and every retire path honours the marker — settlement, the
orphan sweep, recovered-invocation refusals, and the pre-run refusal path. The ownership
duty is discharged durably (`PROJECT_RETIRED` with `project_kept: true`) without
deleting the project, and any persistent sharer makes the shared project undeletable for
its non-persistent siblings. Durable records written before the marker existed fall back
to the immutable stored request (`execution.workspaceRoot` + `access`) at recovery time
(`record_persistent`); rows that carry the key are authoritative and are never
recomputed from a live engine. Older engines keep the legacy snapshot-in-`scope.root`
shape: that host-minted snapshot declares `scope.ephemeral` and registers nothing where
the engine accepts the field (`CLAUDEXOR_EPHEMERAL_SCOPE_MIN_VERSION`), and otherwise
retires its one-shot registration as before.

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
records stay on the marks row and in the chain. A child's copied rows carry the parent's
ready receipts (`engine_imports`, kept only when their request declares exactly the row's
bytes, kind, MIME and name), so the child replays the parent's key and reuses its
resourceId. Every schedule that copies a staged non-image input therefore reads the parent's
custody marks once, whether or not it selects anything, and refuses typed
(`document_import_custody_unavailable`) before the child exists when they cannot be read,
rather than letting a copy mint a fresh import; copies of images alone read nothing.
Scheduling still copies each inherited input's retained staged copy, so a gone copy refuses
the schedule (`SUBTASK_ATTACHMENT_ERROR`) even when its ready receipt could serve the child;
only a direct `delegate_start` replays a ready receipt without that copy. Continuations
inherit no documents: a successor selects again. New stagings get the same Markdown and
DOCX labels on every interpreter; rows recorded earlier with an unprovable MIME (for example
Markdown staged as `application/octet-stream`) keep their label and refuse on routes that do
not declare it.

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
never produce. An `auto` request becomes an ordinary native subagent with a visible marker.

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
and pending work. Saved cognition defers pending replay; other recovery replays
only the invocation's saved binding. Root-direct recovery uses the same path.
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
These rules apply to API-only, subscription-only and mixed installations and to
Advisory, Blocking and Cyber Pro; configured route availability and review floors
remain independent.
