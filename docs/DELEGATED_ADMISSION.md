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
before dispatch. `live` means that Claudexor works in that supplied execution
root; it does not mean direct edits to the authoritative target. The host
captures the result and applies it through its existing integration path.
Ordinary folders use the separate supported direct/copy work-product shape in
`delegate_directory.py`.

One start provisions nothing: a continuation (`continue_from` on `delegate_start`,
`delegate_continuation.py`) of a run whose private snapshot is still undisposed
runs in THAT snapshot, with the same execution root, baseline and binding, and
takes its custody over, so the earlier work is captured once in the successor's
cumulative patch. A predecessor whose patch was already applied or rejected is
continued from a fresh snapshot of the target. The admission floors are unchanged:
the continuation's access is the caller's own, never wider than the run it continues.

The harness can execute model-generated commands under the operator's OS
identity. It is not assumed hostile, but the host cannot review each command
before it runs.

When a private snapshot exists, the host appends a separate typed execution
binding after the immutable inherited work order: the snapshot is the writable
root and the stable project root is a read-only identity until explicit
integration. Legacy engines that expose only the snapshot as `scope.root`
receive the same binding using that root. Full native access does not make the
binding enforceable by itself, so terminal capture records authority-tree drift
as diagnostic evidence. A ready-no-changes result remains a normal no-change
capture even when the shared authority tree moved; the evidence names the
changed paths and keeps authorship unknown. A ready-with-changes result retains
its private artifact and the existing locked apply check decides whether
integration is safe. Excluded nested repositories remain outside the snapshot
inventory and are disclosed as an untracked residual.

A read-only child requests `mode: ask`, `access: readonly` under Claudexor's
ordinary envelope. Its parent-selected starting folder is the session project
root, not a read boundary; omitting it preserves inherited/folderless behavior.
The host reads effective access back for both shapes;
the delegated HOME/boundary checks below apply only to marker-carrying runs.

## 2a. Stable project identity and persistent registration

For Git and skill payload work, fresh mutating starts on an engine satisfying the workspace-root release
contract (`CLAUDEXOR_DELEGATED_WORKSPACE_ROOT_MIN_VERSION = "3.8.1"`) register and retain
the user's actual target project in `scope.root`, while the child's writable filesystem
rides separately as the private snapshot in `execution.workspaceRoot`. That registration
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
snapshot still owns patch delivery, and explicit task constraints remain in force.
The owned gateway grants full access only for an absent scoped trust record,
preserving an existing denial. Older immutable snapshots without an access field
keep workspace_write; retries keep their exact recorded request. Runtime review
sessions and genuinely read-only tasks retain readonly/ask. Scoped HOME and
actual-access receipts remain separate facts, and scoped trust grants persist
without an automatic cleanup policy.

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
  read-only delegation and must be the lowest engine that serves it. Read-only sends no
  `execution` block at all.
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
