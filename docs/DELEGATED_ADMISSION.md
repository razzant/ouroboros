# Delegated-run admission — threat model

Status: **schema/capability floors enforced at admission; the OS containment boundary is read back per attempt and
DISCLOSED, never required.** Owner: `ouroboros/config.py` (the two floors),
`ouroboros/subagents.route_health` (the admission decision),
`ouroboros/gateways/claudexor.attempt_containment` (the applied-fact reader) and
`ouroboros/tools/delegate.py` (the three-place disclosure). This document is the reason
those numbers and that predicate are what they are; change it in the same commit as the code.

Claudexor owns the mirror document, `docs/DELEGATED_CONFINEMENT.md` in its own tree, which
describes the MECHANISM. This one describes what Ouroboros can know about that mechanism from
the outside, and what it says when the answer is "nothing was applied".

## 1. The asset

`~/.claudexor/v3/daemon/token` is a bearer for the ENTIRE `/v2` control API. A process that
reads it can start runs at any access level on any registered project. Every authority
derivation Ouroboros performs — the access profile, the run shape, the write-surface
predicate — is decoration downstream of a child that holds it.

The daemon runs as the operator, so a daemon Ouroboros did not start keeps its token in the
operator's own home, at an absolute path a scoped `HOME` does not redirect. Ouroboros's OWN
daemon (D30) is the exception: it is spawned under `CLAUDEXOR_CONFIG_DIR`, and that override IS
the complete relocatable root — its token lives under `data/claudexor/`, not in the operator's
home. Either way the token is an absolute path the child does not have to guess.

## 2a. Stable project identity and private execution workspace

Fresh mutating delegated starts on an engine satisfying the workspace-root release
contract register and retain the actual target project in `scope.root`. The host copies
that target into a private snapshot and sends the snapshot as `execution.workspaceRoot`,
so the engine's project identity and the child's writable filesystem are separate facts.
The pinned 3.8.0 engine and older engines keep the legacy shape, with the private
snapshot itself in `scope.root`; they do not receive the new field. A retry always
replays its recorded request byte-for-byte. An already accepted idempotency key may
therefore replay across an engine upgrade; an unknown legacy pending start that the
future schema rejects remains pending with a typed compatibility reason, while a
future-shaped pending body is held back from an old engine. Read-only starts keep the legacy
`mode: ask` envelope and send no `execution.workspaceRoot`.

This shape depends on a Claudexor schema that accepts `workspaceRoot`. Ouroboros uses the
explicit release contract `CLAUDEXOR_DELEGATED_WORKSPACE_ROOT_MIN_VERSION = 3.8.1`, the
next compatible semver release carrying Claudexor PR216 after the currently pinned 3.8.0.
Older engines keep the legacy snapshot-in-`scope.root` shape and retire their one-shot
registration; a future engine at or above the contract gets the stable-target shape.

## 2. The actor

A delegated MUTATING child: `mode: agent`, `access: workspace_write`,
`execution.isolation: live`. Claudexor hands the harness a real shell in the nanny's own
worktree. The command stream is model output and the host cannot review each command before it
runs. Not assumed hostile — assumed uncontrolled.

A delegated READ-ONLY child (`mode: ask`, `access: readonly`) is not this actor. It gets no
shell that can mutate, and it stays inside Claudexor's ordinary envelope.

## 3. What Ouroboros actually controls

Only ADMISSION and REPORTING. Ouroboros is an HTTP client of a daemon it does not build, ship,
or version. It cannot confine the child; it can decline to start a run, and it can state
afterwards what the run actually got.

The marginal escalation is worth naming before any defence is priced against it (AGENTS.md
"Name the marginal escalation, not the scary noun"): this child already holds a shell in the
nanny's worktree, running the operator's own code as the operator. The step from "shell" to
"shell plus the daemon token" is real but small, and it does not buy a lane-wide refusal.

So the question is NOT "against which engines is this an acceptable act?" but **"what did this
run actually get, and does everyone downstream know?"**

## 4. The version bands (measured 2026-08-03, not assumed)

Probed live against the operator's running daemon, and read out of the Claudexor tree at
`/Users/anton/Clawdexor` for the bands no local daemon runs.

| Engine | `execution.delegated` | What the child actually gets | Verdict |
| --- | --- | --- | --- |
| ≤ 3.2.x | **400** `invalid_request`, `fieldErrors: {"/execution/delegated": ["Unexpected field; not part of this request."]}` | run never starts | below the marker floor — refused, because it cannot run |
| 3.3.0 – 3.3.1 | accepted | a scoped `HOME` — a CONVENTION. `~`-relative lookups redirect; `/Users/<op>/.claudexor/v3/daemon/token` is read with an absolute path and is READABLE. No confinement fields exist on the attempt record at all | admitted, and reported as UNCONFINED |
| ≥ 3.3.2, macOS | accepted | Seatbelt profile denying the Claudexor runtime tree and the operator credential stores, PROVEN against a denied path before the harness spawns; recorded as `confinement_mechanism` + `confinement_verified_denied_path` | admitted, and reported as CONFINED |
| 3.3.2, elsewhere | accepted | nothing, and the run does not proceed: `applyConfinement` threw `ConfinementUnavailableError` off darwin and the evidence gate refused to terminalize | REFUSED by the engine (`delegated_confinement_unavailable`) |
| ≥ 3.3.3, elsewhere | accepted | nothing enforced. `confinementMechanism()` returns null off darwin and the engine works anyway, disclosing the absence — `docs/DELEGATED_CONFINEMENT.md` §7: "There is no second policy. On every other platform `confinement_mechanism` is null, `confinement_verified_denied_path` is null, and `confinement_unavailable_reason` says why." | admitted, and reported as UNCONFINED |

3.3.3 is where proceed-and-disclose replaced the refusal, not 3.3.6. Between them, 3.3.3–3.3.5
did ship a real Linux bubblewrap boundary; 3.3.6 removed it as an owner decision, leaving the
scoped `HOME` plus a disclosed absence as the whole non-macOS design. None of 3.3.3–3.3.5 was
ever tagged or published, so the band above is what any reachable engine does.

The live 3.2.0 daemon answers the read-only body with nothing but the fake-root error
(`project root does not exist`), i.e. it schema-accepts every field that lane sends. The
mutating body is rejected on the field, before the root is even looked at.

### Why the floor is the MARKER release and not the boundary release

Both were tried. The floor sat at 3.3.2 — the release that added the boundary — on the
reasoning that 3.3.0–3.3.1 write `harness_home_isolated: true` while the token stays readable,
so admitting them would produce a receipt for a confinement that is not there.

That reasoning was right about the receipt and wrong about the remedy. **The last row of the
table is the same defect the floor was supposed to prevent, and the floor cannot see it:** a
3.3.2 build declares 3.3.2 on every host and applies a boundary on one of them. A version
describes a BUILD; it never describes what THIS attempt did. Using it as a proxy for "a
boundary was applied" is false in both directions — it refuses engines that would have been
honestly reported, and it passes hosts where nothing was applied.

The receipt is fixed where the receipt is written (§8), not by narrowing admission. Once the
report tells the truth, the whole band from 3.3.0 up is admissible, and the floor means the
one thing a version can honestly mean: **below 3.3.0 the request is a 400 and no run exists.**

## 5. Why a version at all, and why not a capability probe

For the SCHEMA question a version is the only answer available. Verified rather than assumed:

- `POST /v2/handshake` returns `{protocolMajor, compatible, operationsPath, engine: {version,
  sha, entry}}`. There is no capability list of any kind — checked live, and checked in the
  3.3.2 source, where the handshake responder is unchanged.
- `GET /v2/agent-capabilities` publishes `runControlKeys` derived from **top-level** request
  keys only. `execution` appears; `execution.delegated` is nested and therefore invisible.
  The catalog SCHEMA is field-identical between 3.2.0 and 3.3.2 — the only diff is one
  `.describe()` string, so it gained nothing a probe could read.
- The per-harness `delegation` object in that catalog is about **Claudexor MCP injection** —
  whether the harness can be handed sub-agent tools. It is `available: true` on the live 3.2.0
  daemon, which rejects the marker outright, so reading it as a delegation signal would admit
  precisely the engines that cannot serve the lane.
- A probe by BEHAVIOUR is unavailable: `RunExecution` is `.strict()`, so the only way to learn
  whether the field is accepted is to send it, and sending it on an engine that accepts it
  STARTS THE RUN. There is no dry-run key in `runControlKeys`. The probe and the act are one.

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
- `attempt_containment` — the applied-evidence reader. Not a gate: it decides what is SAID,
  never whether the run happens.

An engine between the two floors serves read-only delegation and refuses mutating delegation.
That is the owner's explicit decision, and it is why the marker floor is not simply raised into
the transport floor.

Both floors fail CLOSED: `engine_at_least` compares an absent or unparsable version as `(0,)`,
below every floor.

Refusal never degrades into metered native execution on a PIN. An explicit `executor="harness"`
request that cannot be served becomes `blocked` — `agent.executor_blocked_outcome` ends the
child unrun — because silently spending API money is the one outcome an explicit pin must
never produce. An `auto` request becomes an ordinary native subagent with a visible marker.

## 7. What this does NOT cover

Stated plainly, because a floor described as total is worse than a narrow one.

- **Not the enforcement.** Ouroboros admits; the engine confines. The floor is a claim about a
  build, checked against a self-reported number, and it is now used only for the schema
  question, where that is enough.
- **Not a lying or downgraded daemon.** The version is self-reported over loopback, and so are
  the applied facts on the attempt record. Anything that can forge either already runs as the
  operator and has the token.
- **Not the gap between two repos.** Ouroboros and Claudexor have no shared build. That the
  release carrying the marker declares ≥ 3.3.0 is a RELEASE GATE on the engine side, not
  something this pin can enforce. It holds without an edit for every bump above the floor and
  fails closed if a release ever breaks it.
- **Not a promise that anything is confined.** A delegated mutating run is allowed on a host
  with no boundary mechanism at all. What is guaranteed is that the run is not DESCRIBED as
  confined when it is not — the disclosure, not the boundary, is the invariant.
- **Not what the boundary itself leaves open where it does exist.** The vendor credential root
  stays readable to the child and the network is not fenced. Those are the engine's to state
  and it states them in `docs/DELEGATED_CONFINEMENT.md` §8. Ouroboros must not re-describe
  them as covered.
- **Not the read-only lane's confinement.** A read-only child is scoped by Claudexor's ordinary
  envelope. Ouroboros asks for no marker and verifies no boundary there.
- **Not an engine that applied a boundary and recorded nothing.** Silence is read as "no
  boundary", so such a run is disclosed as unconfined when it was in fact confined. That is
  the honest limit of an applied-fact reader, and it is the safe direction: the consequence is
  a disclosure, never a refusal.
- **Not free of every harness NAME.** One named residual, disclosed rather than removed:
  `gateway/claudexor_accounts.py::_build_login_request` branches on `harness == "codex"` in
  three places (login setup only — never admission, routing or confinement). The branch is
  load-bearing: `loginFlow` exists only for codex and is a 400 elsewhere, and a non-codex
  login with no explicit transport would default daemon-side to `transport=daemon`, the
  macOS Terminal.app handoff D30 forbids — so `client_pty` is forced instead. It mirrors
  Claudexor's own setup-transport rule, not Ouroboros policy, and deleting it breaks D30.
  It is the ONLY harness-name branch in the core (`ouroboros/`, `supervisor/`, `server.py`,
  `launcher.py`). Removal condition: when the engine makes non-codex logins daemon-hosted,
  the branch goes and this bullet with it.

## 8. Evidence, not intention — and the disclosure it feeds

What the run actually got is read back from the run's own artifacts
(`<runDir>/attempts/<id>/attempt.yaml`). The HOME pair is artifact-only — the engine projects it
onto no `/v2` response — while the boundary is also on the run detail, as
`candidates[].confinement` (`proven` / `mechanism` / `verifiedDeniedPath` / `unavailableReason`,
since 3.3.6); the artifact stays the one reader here because it answers both halves at once.
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
the applied facts, omitted when the attempt died before its home was decided (and an engine
older than 3.3.2 put no applied facts on `attemptFailureRecord` at all). A missing mechanism
collapses to "no boundary", because the consequence there is a DISCLOSURE. Each silence is read
in the direction whose failure mode is recoverable.

**A breach is exactly two facts** (simplified 2026-08-11, Poltergeist phase A3; the
2026-08-07 refinement went one step further): a recorded `harness_home_isolated: false`,
or an applied home EQUAL to the operator's own (the claim is the lie, whatever boundary
sits beside it). A scoped home NESTED under `$HOME` is NOT a breach — with or without a
recorded boundary. The engine roots every scoped home under its own runtime dir, which
lives under `$HOME` on every host it supports, and on a host with no boundary mechanism
(every non-macOS host today) it CANNOT record one — so the former nested-without-mechanism
rule cancelled every mutating Linux run post-factum while the work was already done and
healthy. The boundary-less nested shape flows to the existing disclosed-unconfined path
below instead: the token stays reachable by a relative walk and the disclosure SAYS so,
but the child already holds a shell in this worktree, and cutting the lane on every
boundary-less host costs more than the marginal step it prevents (AGENTS.md "Disclose
instead of forbid"). The engine's typed `confinement_unavailable_reason` — read from the
SAME attempt artifact — rides the disclosure as an amplifier (why this host has no
mechanism); it is telemetry, never an admission token, and its presence never excuses a
recorded FALSE.

Where no boundary was applied, the fact is written LOUDLY into three places (AGENTS.md
"Disclose instead of forbid"):

1. **the durable record** — a `delegate_run_unconfined` event, once per run, carrying the
   note the parent was given, so the forensic trail of an integrated patch says where the
   work came from;
2. **the child's own prompt** — its instructions state that the boundary is a REQUEST and not
   a fact, that it must work as though there is none, and that it must not describe itself as
   sandboxed. It cannot be told which way it went, because nothing at start knows: the engine
   decides per attempt and records the fact afterwards;
3. **the parent-facing result** — `delegate_wait`'s terminal payload carries `containment`
   with `os_boundary`, `verified`, and `disclosed`/`attempts`, and a note naming what was
   reachable rather than merely saying a check failed.
