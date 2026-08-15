"""The PREPARE phase of one dispatch (RWS v2 §3.1 step 2).

Why the protocol is three-phase and not two. The old shape was "look at the
args, then run" — and every guard that needed a fact about the target went and
got it itself, whenever it happened to need it. That produces three failures at
once: the same fact is computed several times and may disagree with itself; the
guard is hard-wired to the Home filesystem, because that is where a `Path` probe
goes; and a remote placement needs an RPC per fact, which is unaffordable.
Splitting a PREPARE phase out fixes all three with one move — the operation's
facts are gathered ONCE, up front, from the target that will actually run the
command, and the guards become placement-blind consumers of that snapshot. The
third phase (EXECUTE) then has nothing left to decide: it receives a token bound
to the exact target-canonical arguments that were authorized, so it cannot drift
from what policy approved, and the target revalidates that binding as execution
INTEGRITY rather than as a second policy authority.

`PreparedCall` is the phase's output and the single object the rest of the
dispatch reads:

* `placement` / `facts` — one placement read per operation, and the fact door
  derived from it (never a second placement authority);
* `executor` — the derived `ExecutorRef` projection for the execute phase;
* `unavailable` — a TYPED refusal when prepare itself could not run: no broker in
  this process, a malformed placement projection, or the target's own refusal
  (which keeps its own code). The operation then stops at the prepare boundary
  instead of being authorized over facts nobody has;
* `operation` — the native operation the EXECUTE phase will run, or `""` for a
  Home-local operation. TWO facts have to hold for it to be non-empty, and this is
  the single place both are consulted. The routing TABLE
  (`tool_capabilities.remote_native_operation_for_tool`) says the target declares a
  counterpart at all, because a tool it never declared must not be sent to it:
  asking the target to prepare `journal_write` would turn a Home faculty into a
  remote refusal. The root MATRIX (`workspace_ref.root_is_target_native`) then says
  this CALL's bytes live on the target — only `active_workspace` does. A remote
  `read_file(root='artifact_store')` is Home-local for the second reason even though
  the first holds;
* `projections` + `prepared_hash` / `prepared_token` / `expires_at_ms` — bound
  AFTER the argv is final (python pre-resolution may still rewrite the
  interpreter), so the token commits to the argv that was actually authorized.

Binding order matters and is the reason prepare and binding are two calls rather
than one: the interpreter FACT must precede argv finalization (a remote
interpreter is a prepare fact, not a Home probe), while the token must follow it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import uuid
from typing import Any, Mapping

from ouroboros.execution_facts import ExecutionFacts, facts_for_ref
from ouroboros.remote_contracts import ContractDriftError
from ouroboros.remote_refusal_actions import ACTION_REPORT_DEFECT
from ouroboros.tool_capabilities import dispatch_resource_root, remote_native_operation_for_tool
from ouroboros.tools.dispatch_args import ArgProjections, ExecutionArgs, project_dispatch_args
from ouroboros.workspace_ref import SshWorkspaceRef, root_is_target_native, workspace_ref_for


@dataclasses.dataclass(frozen=True)
class PreparedCall:
    """Target facts + the token binding of one operation's execution args."""

    tool: str
    placement: str
    operation_id: str
    facts: ExecutionFacts
    executor: Any = None
    # The native operation an ssh dispatch will execute; "" = Home-local tool.
    operation: str = ""
    # The target's own prepared call for an ssh placement: the token binding plus
    # the target-canonical execution args the EXECUTE phase must replay. Local and
    # docker placements have none — Home is the target, so there is nothing to bind
    # across a wire.
    native: Any = None
    # A malformed operator/runtime executor_ref reads as plain NON-COVERAGE, the
    # existing semantics of every process surface (`covers()` treats a mapping
    # failure as "run on the host"). Prepare keeps that behavior and merely makes
    # the malformed ref VISIBLE instead of swallowing it at each call site.
    executor_error: str = ""
    unavailable: str = ""
    # The hash-bound export policy Home decided for this operation (§3.2), carried
    # so the IMPORT side can validate the returned manifest against the very
    # document the target was handed — not against a policy recomputed later, which
    # would silently accept a manifest produced under different rules.
    export_policy: Any = None
    projections: ArgProjections | None = None
    prepared_hash: str = ""
    prepared_token: str = ""

    @property
    def available(self) -> bool:
        return not self.unavailable

    @property
    def bound(self) -> bool:
        return self.projections is not None and bool(self.prepared_token)

    @property
    def native_routed(self) -> bool:
        """Whether the EXECUTE phase runs this operation on the target.

        True only when BOTH halves are in place: the routing table names a native
        operation AND the target actually prepared it. A Home-local tool on an ssh
        task is False here and keeps its Home handler.
        """
        return bool(self.operation) and self.native is not None

    def binds(self, execution_args: ExecutionArgs) -> bool:
        """Whether the token commits to exactly these execution args.

        The Home-side half of the integrity check the target repeats: if the argv
        or the cwd changed after authorization, the binding no longer matches and
        the operation must not run.
        """
        if not self.bound:
            return False
        return _execution_hash(self.tool, execution_args) == self.prepared_hash


@dataclasses.dataclass
class OutstandingPrepare:
    """The prepared operation this dispatch left OUTSTANDING on the target.

    The prepare/execute pair has a third member — `abort_prepared` — and until this
    register existed nothing in production called it: every Home refusal that lands
    AFTER a successful prepare returned its text and left the target holding a
    reserved token and its staged blobs until the TTL expired. Enumerating those
    refusal sites and adding an abort to each would be a list to keep in sync with a
    ~300-line pipeline, so the fact is registered instead: prepare CLAIMS, execute
    RELEASES, and the dispatch boundary withdraws whatever is still held — including
    on refusal paths nobody thought to enumerate, and on an exception.

    It is a per-dispatch object threaded through the call rather than state on the
    registry, because a tool handler can re-enter `ToolRegistry.execute` and two
    dispatches must never share one slot.
    """

    prepared: PreparedCall | None = None

    def claim(self, prepared: PreparedCall) -> None:
        """The target now holds a token (and any staged blobs) for this call."""
        self.prepared = prepared

    def release(self) -> None:
        """Hand-off: the pair is now the target's to close, not Home's to withdraw."""
        self.prepared = None


def _execution_hash(tool: str, execution_args: ExecutionArgs) -> str:
    payload = {"tool": str(tool), **dataclasses.asdict(execution_args)}
    payload["argv"] = list(execution_args.argv)
    payload["paths"] = list(execution_args.paths)
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def prepare_operation(
    ctx: Any,
    tool: str,
    args: Mapping[str, Any] | None = None,
    *,
    outstanding: OutstandingPrepare | None = None,
) -> PreparedCall:
    """Read the placement ONCE and gather the operation's target facts.

    This is the dispatch pipeline's single placement-resolution site: everything
    downstream — the fact door, the executor projection, the guards — is derived
    from this one read, so a second, disagreeing placement cannot exist.

    ``args`` are the caller's arguments as they stand BEFORE Home's own
    projection. An ssh target needs them to know which paths, cwd and interpreter
    the operation is about; it resolves those itself and returns the facts, which
    is why the remote interpreter is a prepare fact rather than a Home probe.

    ``outstanding`` is the dispatch's abort register: a prepare that leaves state on
    the target CLAIMS it here, so a Home refusal further down the pipeline can
    withdraw it instead of letting the token expire (see :class:`OutstandingPrepare`).
    """
    from ouroboros.remote_export_policy import policy_for_operation
    from ouroboros.workspace_diagnostics import RemoteWorkspaceError
    from ouroboros.workspace_executor import (
        SshExecutorUnavailableError,
        executor_ref_from_ctx,
        prepare_native_operation,
    )

    ref = workspace_ref_for(ctx)
    facts = facts_for_ref(ref)
    try:
        executor: Any = executor_ref_from_ctx(ctx)
        executor_error = ""
    except Exception as exc:  # noqa: BLE001 — see PreparedCall.executor_error
        executor, executor_error = None, f"{type(exc).__name__}: {exc}"
    operation = remote_native_operation_for_tool(tool)
    remote = isinstance(ref, SshWorkspaceRef)
    root = dispatch_resource_root(tool, args)
    if remote and operation and not root_is_target_native(ref, root):
        # ROOT MATRIX (ratified Q2а). A routable tool on a remote placement is still
        # not enough: only `active_workspace` has its bytes on the target. A call
        # against a HOME-native root — the task's artifact, the owner's files, task
        # scratch, a skill payload, the system repo — is Home's operation, so it keeps
        # its Home handler exactly as it would under a local placement. This is the
        # NORMAL path for a remote task, not a degradation: those roots are where a
        # remote task's artifacts and scratch actually live.
        operation = ""
    prepared = PreparedCall(
        tool=str(tool),
        placement=facts.placement,
        operation_id=uuid.uuid4().hex[:16],
        facts=facts,
        executor=executor,
        operation=operation,
        executor_error=executor_error,
    )
    if not remote or not operation:
        # Local and docker placements prepare on Home: the fact accessor IS the
        # prepare response, computed lazily and memoized per operation upstream.
        #
        # A HOME-LOCAL operation on a remote task takes the same road and asks the
        # target for NOTHING — whether it is Home-local because the target declares no
        # counterpart (memory, the task tree, the web) or because the call is about a
        # Home-native root (the task's artifact, the owner's files). Its
        # `placement` still reads "ssh" and its fact door is still the remote one —
        # the task's placement is a fact, not a per-tool convenience — and since a
        # Home-local tool never asks for a TARGET fact, the door is simply unused.
        return prepared
    # The export policy is decided HERE, by Home, and travels WITH the prepare —
    # not fetched by the target and not defaulted on the target. Binding it to the
    # operation at the same moment as the facts is what lets Home check the
    # returned manifest against the one document it authorized (§3.2); a policy
    # computed later, or per channel, would be a second policy.
    policy = policy_for_operation(ctx, prepared.operation, workspace_root=ref.remote_root)
    try:
        native = prepare_native_operation(
            prepared.executor,
            prepared.operation,
            args={**dict(args or {}), **policy.arg_payload()},
            task_id=str(getattr(ctx, "task_id", "") or ""),
            operation_id=prepared.operation_id,
        )
    except SshExecutorUnavailableError as exc:
        # No broker in THIS process, which no retry moves — and the model got this as
        # a bare sentence with no next step, so a task that hit it reported an
        # unexplained remote failure to the owner.
        return dataclasses.replace(
            prepared, unavailable=f"{exc} [action: {exc.action}]"
        )
    except ContractDriftError as exc:
        # A contract Home itself could not project. Caught BEFORE the bare
        # `ValueError` arm below (it is a `ValueError` subclass) so it keeps its own
        # code and action instead of being reported as its Python class name — the
        # exact substitution that made the owner's failure unreadable.
        return dataclasses.replace(
            prepared,
            unavailable=f"{exc.code}: {exc} [action: {exc.action}]",
        )
    except ValueError as exc:  # malformed/absent ssh executor projection
        # A Home-side projection that does not typecheck is a wiring defect, not
        # something the owner or a retry can move; the class name alone said neither.
        return dataclasses.replace(
            prepared,
            unavailable=f"{type(exc).__name__}: {exc} [action: {ACTION_REPORT_DEFECT}]",
        )
    except RemoteWorkspaceError as exc:
        # A REAL remote refusal, not a missing build. It keeps its own typed code
        # so the diagnosis says what the target said, and the operation still stops
        # at the prepare boundary instead of being authorized over absent facts.
        #
        # The ACTION rides along when the refusal has one. `unavailable` is a plain
        # string that reaches the model and the transcript verbatim, and this line is
        # why the owner's report read `REMOTE_EXECUTION_UNAVAILABLE: ValueError:
        # export policy has unknown fields: [...]` — a diagnosis with no next step,
        # handed to the one reader who cannot look the code up. Refusals carrying the
        # default `retry` action print exactly as before, so nothing gains noise.
        action = getattr(exc, "action", "retry")
        suffix = f" [action: {action}]" if action and action != "retry" else ""
        return dataclasses.replace(prepared, unavailable=f"{exc.code}: {exc}{suffix}")
    # The fact door is now backed by the bundle the target just filled — the same
    # single read that produced the token the EXECUTE phase will replay.
    returned_hash = str((getattr(native, "native_facts", {}) or {}).get("export_policy_hash") or "")
    if returned_hash != policy.policy_hash:
        # The target either dropped the policy or normalized it into something
        # else. Either way it is about to build blobs under rules Home cannot name,
        # so the operation stops at the prepare boundary — the same place an absent
        # fact stops it.
        return dataclasses.replace(
            prepared,
            unavailable=(
                f"REMOTE_EXPORT_POLICY_UNBOUND: the target prepared "
                f"{prepared.operation!r} under policy {returned_hash[:16] or '<none>'} "
                f"but Home authorized {policy.policy_hash[:16]}"
            ),
        )
    prepared = dataclasses.replace(
        prepared,
        native=native,
        export_policy=policy,
        facts=facts_for_ref(ref, getattr(native, "native_facts", {}) or {}),
    )
    if outstanding is not None:
        # The target now holds a token and whatever blobs the prepare staged. Every
        # `unavailable` return above skipped this on purpose: those either never
        # reached the target or are the target's OWN refusal, so there is no Home-side
        # reservation to withdraw.
        outstanding.claim(prepared)
    return prepared


# The model-facing prefix of an UNDISCLOSED argv substitution. Distinct from the
# binding mismatch (which catches a rewrite between authorize and execute on HOME):
# this one catches the target proposing a different command than Home was asked for.
REMOTE_ARGV_SUBSTITUTED = "⚠️ REMOTE_ARGV_SUBSTITUTED"
# Its path-argument sibling. A separate code because it is a different fact and the
# reader needs to know which: "the target wants to run another command" and "the
# target wants to touch another file" are diagnosed differently.
REMOTE_PATH_SUBSTITUTED = "⚠️ REMOTE_PATH_SUBSTITUTED"
# Which argument of each tool the TARGET rewrites when it canonicalizes a command.
_TARGET_ARGV_KEYS: dict[str, str] = {
    "run_command": "cmd",
    "start_service": "cmd",
    "verify_and_record": "cmd",
    "run_script": "interpreter",
}
# Which argument of each tool the TARGET canonicalizes as a PATH. Same table shape
# as the argv one, and for the same reason: the target rewrites this key during
# prepare (`workspace_native.prepare_native_operation` normalizes the relative
# spelling), so Home has to know where to look to compare.
_TARGET_PATH_KEYS: dict[str, str] = {
    "read_file": "path",
    "write_file": "path",
    "edit_text": "path",
    "list_files": "path",
    "search_code": "path",
    "query_code": "path",
}


def _target_argv(prepared: PreparedCall) -> tuple[str, ...]:
    """The command the TARGET intends to execute, as it canonicalized it."""

    key = _TARGET_ARGV_KEYS.get(prepared.tool, "")
    raw = (getattr(prepared.native, "execution_args", {}) or {}).get(key)
    if isinstance(raw, list):
        return tuple(str(item) for item in raw)
    return (str(raw),) if raw else ()


def _home_argv(prepared: PreparedCall, args: Mapping[str, Any]) -> tuple[str, ...]:
    """The command HOME is about to authorize, in the same shape."""

    key = _TARGET_ARGV_KEYS.get(prepared.tool, "")
    raw = args.get(key)
    if isinstance(raw, list):
        return tuple(str(item) for item in raw)
    return (str(raw),) if raw else ()


def reconcile_target_argv(prepared: PreparedCall, args: dict[str, Any]) -> tuple[str, str]:
    """Make the argv Home AUTHORIZES the argv the target will EXECUTE.

    The target canonicalizes a command during prepare: it resolves an interpreter, and
    it autocorrects a shell-quoting mistake (`grep 'a\\|b'` → `grep -E 'a|b'`, disclosed
    in ``autocorrect_note``). Home was authorizing its own spelling and the target was
    executing its own — so the guards approved one command and a different one ran.
    Nothing detected it: the prepared token binds Home's hash and the target revalidates
    against ITS args, so both sides agreed with themselves.

    Two outcomes, split by whether the target was HONEST about the rewrite, because
    those are two different facts:

    * DISCLOSED (an autocorrect note, or an interpreter resolution the facts name) —
      the target's argv is adopted into ``args`` HERE, before the guard projection, so
      the full pipeline authorizes the command that will actually run. That is stricter
      than the local placement, where the same autocorrect happens inside the handler
      after the guards have seen the original. The disclosure travels to the model.
    * UNDISCLOSED — a refusal. A target that rewrites a command without saying so has
      no honest reason to, and "authorized one thing, ran another" is exactly the class
      of failure the three-phase protocol exists to make impossible.

    Returns ``(disclosure_note, refusal)``; ``args`` is mutated in place on adoption so
    the guard projection, the token binding and the execute-time re-projection all read
    one set.
    """

    if not prepared.native_routed and prepared.native is None:
        return "", ""
    key = _TARGET_ARGV_KEYS.get(prepared.tool, "")
    if not key:
        return "", ""
    target, home = _target_argv(prepared), _home_argv(prepared, args)
    if not target or target == home:
        return "", ""
    facts = getattr(prepared.native, "native_facts", {}) or {}
    note = str(facts.get("autocorrect_note") or "").strip()
    interpreter = str(facts.get("interpreter") or "")
    interpreter_only = (
        bool(interpreter)
        and len(target) == len(home)
        and target[:1] == (interpreter,)
        and target[1:] == home[1:]
    )
    if not note and not interpreter_only:
        return "", (
            f"{REMOTE_ARGV_SUBSTITUTED}: the target prepared a different command than "
            f"Home authorized and disclosed no reason for it. Authorized "
            f"{list(home)!r}; the target intended {list(target)!r}. The process was not "
            "started."
        )
    args[key] = list(target) if isinstance(args.get(key), list) or len(target) > 1 else target[0]
    if not note:
        # An interpreter resolution is already attested through the python-resolution
        # record; re-authorizing it silently would be the quiet acceptance this refuses.
        return "", ""
    return (
        f"ℹ️ REMOTE_ARGV_AUTOCORRECTED: the target rewrote this command and said so; "
        f"Ouroboros re-authorized the rewritten form through the full guard pipeline "
        f"before running it. {note.strip()}"
    ), ""


def reconcile_target_paths(prepared: PreparedCall, args: dict[str, Any]) -> tuple[str, str]:
    """Make the path Home AUTHORIZES the path the target will TOUCH.

    The argv sibling of this function existed and this one did not, so path-bearing
    tools had the exact hole the prepared token was built to close: `ExecutionArgs`
    carried no path field, the execution hash was therefore a CONSTANT per tool, and
    `execute_prepared` replays the TARGET's own ``execution_args``. A target that
    answered a prepare with a different `path` was authorized for one file and ran on
    another, with nothing anywhere disagreeing — Home authorized `write_file
    path="ok.txt"` and `PWNED.txt` appeared; `edit_text path="hello.txt"` edited
    `victim.txt`; `read_file path="hello.txt"` returned `SECRET.txt`.

    The split between accepted and refused is not "did the target say so" but
    something stronger, available here and nowhere else: Home REPRODUCES the
    canonicalization. `native_relative_spelling` is the contract both sides share, so
    a target path that equals Home's own canonical spelling is a tidied spelling and
    is adopted; anything else is a substitution and the operation does not run. No
    disclosure a target could write buys it a different file.

    Returns ``(disclosure_note, refusal)``; ``args`` is mutated in place on adoption
    so the guard projection, the token binding and the execute-time re-projection all
    read one set.
    """

    from ouroboros.workspace_native_contract import native_relative_spelling

    if not prepared.native_routed and prepared.native is None:
        return "", ""
    key = _TARGET_PATH_KEYS.get(prepared.tool, "")
    if not key:
        return "", ""
    target = str((getattr(prepared.native, "execution_args", {}) or {}).get(key) or "")
    home = str(args.get(key) or "")
    if not target or target == home:
        return "", ""
    if not home:
        # Home named no path at all. Nothing is adopted — filling the argument in
        # would turn a public-schema refusal into an executed operation, and that
        # refusal's precedence is contract. The only spelling the target may have
        # substituted for an absent one is the contract's own default; anything else
        # is still a substitution and is still refused.
        if target == native_relative_spelling(None):
            return "", ""
        return "", (
            f"{REMOTE_PATH_SUBSTITUTED}: Home named no path and the target supplied "
            f"{target!r}. The operation was not run."
        )
    try:
        canonical = native_relative_spelling(home)
    except ValueError:
        canonical = ""
    if target != canonical:
        return "", (
            f"{REMOTE_PATH_SUBSTITUTED}: the target prepared a different path than Home "
            f"authorized. Authorized {home!r}; the target intended {target!r}. The "
            "operation was not run."
        )
    args[key] = target
    # A pure spelling normalization Home can derive itself is not news for the model,
    # and announcing it on every `./x` would train the reader to ignore the channel
    # the genuine substitutions arrive on. The token still binds the adopted value.
    return "", ""


REMOTE_ARGS_SUBSTITUTED = "⚠️ REMOTE_ARGS_SUBSTITUTED"


def reconcile_residual_target_args(
    prepared: PreparedCall, args: dict[str, Any]
) -> tuple[str, str]:
    """Refuse a target rewrite of any argument that is NOT canonicalizable.

    The two reconcilers above each name the key their target is allowed to rewrite,
    and enumerating permissions one key at a time is how the path key came to be
    missing for six tools. So the residue is closed instead of enumerated: the target
    may canonicalize its argv, its path and its cwd, and every OTHER argument it hands
    back must be the one Home authorized. `edit_text`'s `old_str`/`new_str` and
    `write_file`'s `content` and `files[]` rows are covered by this and by nothing
    else — a target rewriting those decides what gets written while Home's guards
    approved something else, which is the same defect as the substituted path with a
    different argument name.
    """

    if not prepared.native_routed and prepared.native is None:
        return "", ""
    target_args = getattr(prepared.native, "execution_args", {}) or {}
    canonicalizable = {
        _TARGET_ARGV_KEYS.get(prepared.tool, ""),
        _TARGET_PATH_KEYS.get(prepared.tool, ""),
        "cwd",
    }
    divergent = sorted(
        key
        for key, value in target_args.items()
        if not str(key).startswith("_")
        and str(key) not in canonicalizable
        and json.dumps(value, sort_keys=True, default=str)
        != json.dumps(args.get(key), sort_keys=True, default=str)
    )
    if not divergent:
        return "", ""
    return "", (
        f"{REMOTE_ARGS_SUBSTITUTED}: the target prepared different arguments than Home "
        f"authorized for {', '.join(divergent)}. The operation was not run."
    )


def reconcile_target_args(prepared: PreparedCall, args: dict[str, Any]) -> tuple[str, str]:
    """Reconcile every argument class the target canonicalizes: argv, path, residue.

    One call site in the dispatcher, so a tool that bears several kinds cannot have
    one of them reconciled and the others forgotten — which is exactly how the path
    half came to be missing.
    """

    note, refusal = reconcile_target_argv(prepared, args)
    if refusal:
        return note, refusal
    path_note, refusal = reconcile_target_paths(prepared, args)
    if refusal:
        return note, refusal
    _, refusal = reconcile_residual_target_args(prepared, args)
    return "\n".join(part for part in (note, path_note) if part), refusal


def bind_execution_args(
    prepared: PreparedCall,
    ctx: Any,
    args: Mapping[str, Any],
    *,
    runtime_mode: str = "",
    binding: Any = None,
) -> PreparedCall:
    """Project the FINAL arguments and bind the token to them.

    Called after python pre-resolution, so the hash commits to the argv the
    guards are about to authorize and the executor is about to run — not to an
    earlier draft of it.

    Under a native route the cwd comes from the TARGET's own prepare rather than
    from a Home resolver: it is the one canonicalization that actually describes
    where the process will run, and there is no second read of the placement to
    obtain it. On a Home route it comes from ``binding``, the dispatch's one
    Home-side resolution — and the SAME binding must reach the re-projection that
    checks this token before execute, or the two would project different cwds and
    the integrity check would refuse a call nothing actually rewrote.
    """
    projections = project_dispatch_args(
        ctx,
        prepared.tool,
        args,
        runtime_mode=runtime_mode,
        facts=prepared.facts,
        target_cwd=native_execution_cwd(prepared),
        binding=binding,
    )
    prepared_hash = _execution_hash(prepared.tool, projections.execution_args)
    return dataclasses.replace(
        prepared,
        projections=projections,
        prepared_hash=prepared_hash,
        prepared_token=f"{prepared.operation_id}.{prepared_hash[:32]}",
    )


def native_execution_cwd(prepared: PreparedCall) -> str:
    """The target-canonical process cwd of a native route, else ``""``.

    ``""`` means "resolve on Home", which is what every local/docker dispatch and
    every Home-local tool wants. For a native route the answer is the target's
    own resolution, read from the operation's single prepare — the `execution_args`
    spelling first, because that is the one the target will actually chdir into.
    """
    if not prepared.native_routed:
        return ""
    native = prepared.native
    for source in (
        getattr(native, "execution_args", {}) or {},
        getattr(native, "native_facts", {}) or {},
    ):
        for key in ("cwd", "resolved_cwd"):
            text = str(source.get(key) or "").strip()
            if text:
                return text
    return ""


__all__ = [
    "REMOTE_ARGS_SUBSTITUTED",
    "REMOTE_ARGV_SUBSTITUTED",
    "REMOTE_PATH_SUBSTITUTED",
    "OutstandingPrepare",
    "PreparedCall",
    "bind_execution_args",
    "native_execution_cwd",
    "prepare_operation",
    "reconcile_residual_target_args",
    "reconcile_target_args",
    "reconcile_target_argv",
    "reconcile_target_paths",
]
