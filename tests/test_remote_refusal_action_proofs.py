"""Every refusal that names an action is proven by TAKING that action.

The Guard Proof Rule (docs/DEVELOPMENT.md) says a guard that forbids something must
be shown REFUSING something. This is its twin for the other half of a refusal: a
refusal that ADVISES something must be shown that the advice WORKS. Passing a
correct-looking string proves nothing — the sentence that produced this module was
correct-looking, was rendered to the owner, and was a dead end:

    No bootstrapped healthy connections — run Bootstrap (or Test to refresh
    health) in Settings → Connections

Measured live: with an executor built against an older Home↔execd contract set,
`Test` returned `ok` with `health_fresh: true` and the connection stayed
unselectable, because only Bootstrap writes the contract-set stamp the picker reads.
The owner pressed an offered action, it succeeded, and nothing changed. Call the
family a **dead-end refusal**, and note what it costs: the surface teaches the owner
that its own advice is noise.

Four proofs, in the order they close the class:

`test_every_blocking_state_is_removed_by_the_action_it_proposes` is the load-bearing
one. `BLOCKING_STATES` brings a real connection store into each state a remote
placement can be blocked by; the test then reads the SERVER's own proposed action
off the blocker, looks that action up in `CURES`, performs it, and demands progress —
the same blocker must be gone, and the row must be either selectable or blocked by
something strictly LATER in the removal ladder. Keying the cure by the PROPOSED
action is what makes a dead end fail: had the execd-outdated hint said
`test_connection`, this test would run a health probe and find the same blocker still
there.

`test_no_refusal_code_proposes_an_action_its_own_taxonomy_calls_useless` cross-checks
the register against an INDEPENDENT authority that already existed:
`cli_connections._UNSERVABLE_CODES` is what the owner CLI maps to exit 4, "retrying
will not help". A code in that set whose action is `retry` is a self-contradiction,
and there were fourteen before this branch.

`test_every_typed_remote_refusal_has_a_registered_action` reads every
`RemoteWorkspaceError` code out of the ASTs, so a code added later with no entry
fails here rather than silently defaulting to `retry`.

`test_a_row_with_no_blocker_is_exactly_a_selectable_row` pins the ladder to the
browser's selectability predicate over the whole truth table, because two authorities
that agree by habit are one bug away from disagreeing.
"""

from __future__ import annotations

import ast
import itertools
import pathlib

import pytest

from ouroboros.remote_contracts import CONTRACT_SET_VERSION
from ouroboros.remote_refusal_actions import (
    ACTION_RETRY,
    BLOCK_DEGRADED,
    BLOCK_EXECD_OUTDATED,
    BLOCK_HEALTH_STALE,
    BLOCK_HOST_IDENTITY,
    BLOCK_NOT_BOOTSTRAPPED,
    BLOCK_NOT_READY,
    BLOCK_RETIRED,
    CONNECTION_BLOCKER_HINTS,
    CONNECTION_BLOCKER_ORDER,
    REFUSAL_ACTIONS,
    connection_blocker,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
THIS_BUILD = "6.87.6-proofs"


# ── the state machine under test ─────────────────────────────────────────────
# A projected connection row, assembled the way `api_connections_list` assembles
# one: the store record, then the live broker status, then Home's own evidence.
def _projected(path, connection_id, *, status="unknown", error_code=""):
    from ouroboros.connection_store import get_connection
    from ouroboros.gateway.connections import _runtime_evidence_fields

    row = dict(get_connection(connection_id, path, include_retired=True) or {})
    evidence = _runtime_evidence_fields(path, connection_id, row)
    merged = {**row, "status": status, "error_code": error_code, **evidence}
    # The evidence may carry its own observed status (a fresh health probe), and it
    # wins over the caller's stand-in exactly as it does in the gateway.
    return merged


def _bootstrap(path, connection_id, *, contract_set=CONTRACT_SET_VERSION, host="host-a"):
    """What a SUCCESSFUL Bootstrap does, all three parts of it.

    It pins the observed host identity, records the durable bootstrap claim WITH the
    contract set it installed, and leaves a fresh health observation behind — because
    it just talked to the target. Modelling only the middle part would let a cure
    look like it worked while leaving `health_fresh` false.
    """
    from ouroboros.connection_store import pin_connection_host, record_bootstrap
    from ouroboros.gateway.connections import _record_runtime_health

    pin_connection_host(connection_id, host, path=path)
    record_bootstrap(
        connection_id, build=THIS_BUILD, contract_set=contract_set, path=path
    )
    _record_runtime_health(path, connection_id, {"status": "ready"})


def _test_connection(path, connection_id, *, status="ready"):
    """What a Test does, and ONLY what it does: refresh the health observation."""
    from ouroboros.gateway.connections import _record_runtime_health

    _record_runtime_health(path, connection_id, {"status": status})


# ── the states, and how to get into each one ─────────────────────────────────
# Each entry returns (connection_id, projection_kwargs). The store is real; nothing
# here fabricates evidence a surface would not have.
def _state_retired(path):
    from ouroboros.connection_store import add_connection, retire_connection

    row = add_connection(name="Old box", ssh_alias="old", path=path)
    _bootstrap(path, row["id"])
    retire_connection(row["id"], path=path, has_active_lease=False)
    return row["id"], {"status": "disconnected"}


def _state_host_identity_changed(path):
    from ouroboros.connection_store import add_connection

    row = add_connection(name="Swapped box", ssh_alias="swapped", path=path)
    _bootstrap(path, row["id"], host="host-a")
    # The live probe found a different machine answering the alias; the gateway
    # rewrites the answer with this code (`_connection_action`).
    return row["id"], {"status": "degraded", "error_code": "host_identity_changed"}


def _state_never_bootstrapped(path):
    from ouroboros.connection_store import add_connection

    row = add_connection(name="New box", ssh_alias="new", path=path)
    return row["id"], {"status": "unknown"}


def _state_execd_outdated(path):
    """THE state the owner reported: bootstrapped, healthy, and unusable."""
    from ouroboros.connection_store import add_connection

    row = add_connection(name="Stale box", ssh_alias="stale", path=path)
    _bootstrap(path, row["id"], contract_set=CONTRACT_SET_VERSION - 1)
    return row["id"], {"status": "ready"}


def _state_degraded(path):
    from ouroboros.connection_store import add_connection

    row = add_connection(name="Sick box", ssh_alias="sick", path=path)
    _bootstrap(path, row["id"])
    _test_connection(path, row["id"], status="degraded")
    return row["id"], {"status": "degraded"}


def _state_health_stale(path):
    """Bootstrapped in a PREVIOUS run: the durable half survives, freshness cannot."""
    from ouroboros.connection_store import pin_connection_host, record_bootstrap
    from ouroboros.connection_store import add_connection

    row = add_connection(name="Restarted box", ssh_alias="restarted", path=path)
    pin_connection_host(row["id"], "host-a", path=path)
    record_bootstrap(
        row["id"], build=THIS_BUILD, contract_set=CONTRACT_SET_VERSION, path=path
    )
    return row["id"], {"status": "unknown"}


def _state_not_ready(path):
    """A probe that ANSWERED and reported neither ready nor a fault.

    Reachable only this way: `_record_runtime_health` accepts exactly four statuses,
    so a row with fresh health is `ready`, `degraded`, `disconnected` or `unknown`,
    and one with no fresh health is caught a rung earlier. The rung this replaced
    said "still connecting" and could not be brought about at all — which is how an
    unreachable hint looks from the outside, and why this test insists every rung
    have a state.
    """
    from ouroboros.connection_store import add_connection

    row = add_connection(name="Quiet box", ssh_alias="quiet", path=path)
    _bootstrap(path, row["id"])
    _test_connection(path, row["id"], status="unknown")
    return row["id"], {"status": "unknown"}


#: state → (how to get there, the blocker it must report). The expected code is
#: written down so a ladder reordering that silently changes which blocker is
#: reported fails here instead of quietly changing the owner's advice.
BLOCKING_STATES = {
    BLOCK_RETIRED: _state_retired,
    BLOCK_HOST_IDENTITY: _state_host_identity_changed,
    BLOCK_NOT_BOOTSTRAPPED: _state_never_bootstrapped,
    BLOCK_EXECD_OUTDATED: _state_execd_outdated,
    BLOCK_DEGRADED: _state_degraded,
    BLOCK_HEALTH_STALE: _state_health_stale,
    BLOCK_NOT_READY: _state_not_ready,
}

#: The PROPOSED action → what performing it actually does. Looked up by the action
#: the server proposed, never by the state, which is the whole mechanism: an action
#: that cannot move a state gets executed against it and the state survives.
#:
#: Each cure returns the projection kwargs that describe the world AFTER it ran, so
#: an action that changes what the transport would now report says so.
CURES = {
    "bootstrap_connection": lambda path, cid: (_bootstrap(path, cid), {"status": "ready"})[1],
    "test_connection": lambda path, cid: (_test_connection(path, cid), {"status": "ready"})[1],
    "retrust_host": lambda path, cid: (_retrust(path, cid), {"status": "unknown"})[1],
    "repair_remote_host": lambda path, cid: (
        _test_connection(path, cid), {"status": "ready"}
    )[1],
    "use_active_connection": lambda path, cid: (_replace_connection(path), {"status": "ready"})[1],
}


def _retrust(path, connection_id):
    from ouroboros.connection_store import retrust_connection
    from ouroboros.gateway.connections import _drop_runtime_evidence

    retrust_connection(connection_id, "host-b", path=path, has_active_lease=False)
    # The gateway drops the process-local health with the trust pin: evidence about
    # the machine that used to answer says nothing about the one that does now.
    _drop_runtime_evidence(path, connection_id)


def _replace_connection(path):
    """The only cure for a retired connection: point the work at an active one."""
    from ouroboros.connection_store import add_connection

    row = add_connection(name="Replacement", ssh_alias="replacement", path=path)
    _bootstrap(path, row["id"], host="host-c")
    return row["id"]


@pytest.mark.parametrize("expected_code", sorted(BLOCKING_STATES))
def test_every_blocking_state_is_removed_by_the_action_it_proposes(
    tmp_path, expected_code
):
    """Bring it about, read the advice, TAKE the advice, and require progress.

    The refusal is not allowed to be the judge of its own remedy: the cure is looked
    up by the action the server proposed, so an action that cannot move this state
    gets run against it and the state is still there at the assertion. That is
    exactly what a `Test` proposed for an outdated executor would do.

    "Progress" rather than "cleared" because two states genuinely need two owner
    steps — a replaced host identity must be retrusted and then bootstrapped — and
    the honest requirement is that the ladder never loops: the reported blocker must
    change, and it must change to something strictly closer to ready.
    """
    path = tmp_path / "state" / "remote_connections.json"
    connection_id, projection = BLOCKING_STATES[expected_code](path)

    before = connection_blocker(_projected(path, connection_id, **projection))
    assert before, "the state was supposed to BLOCK; nothing to prove otherwise"
    assert before["blocked_by"] == expected_code, before
    assert before["blocker_hint"] == CONNECTION_BLOCKER_HINTS[expected_code]
    assert before["blocker_rank"] == CONNECTION_BLOCKER_ORDER.index(expected_code)

    proposed = before["blocker_action"]
    assert proposed in CURES, (
        f"{expected_code} proposes {proposed!r}, which this suite cannot perform — "
        "an action no test can take is an action nobody has checked removes anything"
    )
    after_projection = CURES[proposed](path, connection_id)
    # `use_active_connection` cannot heal the row it was offered on: retiring is
    # terminal. Its promise is that a selectable connection now EXISTS, which is the
    # only thing the New Project picker was ever asking.
    if expected_code == BLOCK_RETIRED:
        from ouroboros.connection_store import list_connections

        selectable = [
            row["id"]
            for row in list_connections(path, include_retired=False)
            if not connection_blocker(
                _projected(path, row["id"], **after_projection)
            )
        ]
        assert selectable, "the proposed action produced no usable connection"
        return

    after = connection_blocker(_projected(path, connection_id, **after_projection))
    if not after:
        return
    assert after["blocked_by"] != expected_code, (
        f"DEAD END: {expected_code} proposed {proposed!r}, it succeeded, and the "
        f"same block is still there — {after['blocker_hint']}"
    )
    assert after["blocker_rank"] > before["blocker_rank"], (
        f"{expected_code} proposed {proposed!r} and moved the owner BACKWARDS in the "
        f"ladder, to {after['blocked_by']}"
    )


def test_no_refusal_code_proposes_an_action_its_own_taxonomy_calls_useless():
    """`retry` and exit 4 are a contradiction, and it can be checked mechanically.

    `cli_connections._UNSERVABLE_CODES` is an INDEPENDENT authority: it is the set
    the owner CLI maps to exit 4, which the module's own docstring defines as
    "retrying will not help and no owner action will fix it either". A code in that
    set whose action is `retry` is the dead-end class stated twice in one codebase,
    once correctly. Fourteen codes were in exactly that position.
    """
    from ouroboros.cli_connections import _UNSERVABLE_CODES

    contradictions = sorted(
        code
        for code in _UNSERVABLE_CODES
        if REFUSAL_ACTIONS.get(code, ACTION_RETRY) == ACTION_RETRY
    )
    assert not contradictions, (
        "these codes are mapped to exit 4 (a retry cannot help) and still propose "
        f"`retry`: {contradictions}"
    )
    unregistered = sorted(set(_UNSERVABLE_CODES) - set(REFUSAL_ACTIONS))
    assert not unregistered, (
        "the CLI knows these are unservable but no action is registered for them, so "
        f"the surface can only fall back to `retry`: {unregistered}"
    )


def _literal_refusal_codes():
    """Every refusal code raised with a literal or a module-level constant.

    Read out of the ASTs rather than a hand-kept list, for the reason the packaging
    proofs give: an inventory maintained by hand is an inventory that goes stale the
    first time somebody adds a raise.
    """
    codes: dict[str, list[str]] = {}
    for path in sorted((REPO / "ouroboros").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        constants = {
            target.id: node.value.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name not in {"RemoteWorkspaceError", "_error"}:
                continue
            first = node.args[0]
            code = (
                first.value
                if isinstance(first, ast.Constant) and isinstance(first.value, str)
                else constants.get(getattr(first, "id", ""))
            )
            if isinstance(code, str) and code:
                codes.setdefault(
                    code.strip().lower(),
                    [],
                ).append(f"{path.relative_to(REPO)}:{node.lineno}")
    return codes


def test_every_typed_remote_refusal_has_a_registered_action():
    """A code with no entry defaults to `retry`, which is how this class started.

    Driven by the AST, so this fails for a code added after today with no decision
    about what cures it — the same shape as
    `test_execd_packaging_guard_proofs.py::test_every_declared_prohibition_is_proven_or_reasoned`,
    and for the same reason: the alternative is a list somebody has to remember.
    """
    codes = _literal_refusal_codes()
    assert len(codes) > 30, f"the AST sweep found only {len(codes)} codes — it broke"
    missing = {
        code: sites for code, sites in sorted(codes.items()) if code not in REFUSAL_ACTIONS
    }
    assert not missing, (
        "these typed refusals reach the owner with no registered action, so every "
        f"surface renders `retry` for them: {missing}"
    )


def test_the_register_only_names_actions_a_surface_can_offer():
    """An action nobody implements is a dead end spelled differently."""
    from ouroboros import remote_refusal_actions as registry

    vocabulary = {
        value
        for name, value in vars(registry).items()
        if name.startswith("ACTION_") and isinstance(value, str)
    }
    unknown = sorted(set(REFUSAL_ACTIONS.values()) - vocabulary)
    assert not unknown, f"actions outside the declared vocabulary: {unknown}"
    # The ladder's own codes must be registered AND have an owner sentence, or a
    # blocked connection would be reported with a blank hint.
    for code in CONNECTION_BLOCKER_ORDER:
        assert code in REFUSAL_ACTIONS, code
        assert CONNECTION_BLOCKER_HINTS.get(code), code
    assert set(CONNECTION_BLOCKER_HINTS) == set(CONNECTION_BLOCKER_ORDER)
    assert set(BLOCKING_STATES) == set(CONNECTION_BLOCKER_ORDER), (
        "every blocker in the ladder needs a state that produces it; an unreachable "
        "rung is a hint nobody has ever seen"
    )


def test_the_execd_outdated_hint_refuses_to_offer_the_action_that_cannot_help():
    """The measured dead end, pinned as text.

    Not a paraphrase test: this is the exact confusion the owner hit. The hint for a
    stale executor must name Bootstrap, and must say IN SO MANY WORDS that Test will
    look like it worked and will not help — because the previous sentence offered
    Test as an alternative and the owner took it.
    """
    hint = CONNECTION_BLOCKER_HINTS[BLOCK_EXECD_OUTDATED]
    assert "Bootstrap" in hint
    assert "Test" in hint and "change nothing" in hint
    assert REFUSAL_ACTIONS[BLOCK_EXECD_OUTDATED] == "bootstrap_connection"
    # …and the health-stale hint, whose cure genuinely IS Test, must not send the
    # owner to Bootstrap: an over-broad remedy is the other half of the same defect.
    stale = CONNECTION_BLOCKER_HINTS[BLOCK_HEALTH_STALE]
    assert "Test" in stale and "Bootstrap" not in stale


def test_a_row_with_no_blocker_is_exactly_a_selectable_row():
    """The ladder and the browser's selectability predicate are ONE rule.

    Enumerated over the whole truth table rather than spot-checked, because the two
    live in different languages and "they agree on the cases someone thought of" is
    how a picker starts offering a host admission will refuse.
    """
    for lifecycle, status, compatible, fresh, outdated in itertools.product(
        ("active", "retired"),
        ("ready", "degraded", "disconnected", "connecting", "unknown"),
        (True, False),
        (True, False),
        (True, False),
    ):
        row = {
            "lifecycle": lifecycle,
            "status": status,
            "bootstrap_compatible": compatible,
            "health_fresh": fresh,
            "execd_outdated": outdated,
        }
        # The browser's rule, transcribed from `connections_ui.isSelectableRemoteConnection`.
        selectable = (
            lifecycle == "active"
            and status == "ready"
            and compatible is True
            and fresh is True
            and outdated is not True
        )
        assert bool(connection_blocker(row)) is (not selectable), row


def test_an_unreachable_target_reports_the_action_its_own_refusal_named():
    """The action must survive the admission wrapper, which is where it used to die.

    `workspace_admission` flattens every broker refusal into
    `RemoteWorkspaceUnavailableError`, and both HTTP callers then answered a
    hardcoded `bootstrap_connection` — right for a stale executor, a dead end for a
    changed host identity or a stale bundle.
    """
    from ouroboros.workspace_admission import RemoteWorkspaceUnavailableError
    from ouroboros.workspace_diagnostics import RemoteWorkspaceError

    for code, expected in (
        ("host_identity_mismatch", "retrust_host"),
        ("execd_bundle_contract_outdated", "rebuild_execd_bundle"),
        ("capability_mismatch", "bootstrap_connection"),
        ("remote_platform_unsupported", "choose_supported_host"),
    ):
        cause = RemoteWorkspaceError(code, "refused", phase="bootstrap")
        assert cause.action == expected, code
        wrapped = RemoteWorkspaceUnavailableError("target refused", cause=cause)
        assert wrapped.action == expected, (
            f"the wrapper lost {code}'s action; the HTTP surface would guess again"
        )
    # With no cause at all it still refuses to invent Bootstrap: an absent broker is
    # a restart, and that is what the register says.
    assert RemoteWorkspaceUnavailableError("no broker").action == "restart_ouroboros"


#: Modules that answer an owner with an ``action`` field. The sweep below reads every
#: literal one of them assigns and demands it be a declared vocabulary member, because
#: an ad-hoc string is how one condition came to be called `rebind_project` at two
#: seams and `choose_active_connection` at a third, and one action came to be called
#: both `retry_reconnect` and `reconnect_connection`.
ACTION_EMITTING_MODULES = (
    "ouroboros/gateway/connections.py",
    "ouroboros/gateway/projects.py",
    "ouroboros/gateway/tasks.py",
    "ouroboros/cli_connections.py",
    "ouroboros/remote_worker_proxy.py",
    "ouroboros/remote_contracts.py",
    "ouroboros/remote_ssh_bootstrap.py",
    "ouroboros/execd_state.py",
    "ouroboros/workspace_admission.py",
    # The owner gate in front of the whole surface: its refusals reach the same browser
    # code that renders every other `action`, so it shares the vocabulary.
    "ouroboros/server_auth.py",
)
#: `argparse`'s own keyword collides with the wire field name; it is not an owner
#: action and never reaches a response body.
ARGPARSE_ACTIONS = frozenset({"store_true", "store_false", "append", "count"})


def test_no_surface_invents_an_action_name_outside_the_vocabulary():
    """One vocabulary, or a surface offers a button that does not exist.

    Two spellings of one action are indistinguishable from two different actions to
    the reader, and this codebase had two such pairs. Driven by the AST so a literal
    added later has to be a decision rather than a typo.

    BOUNDARY — this reads SOURCE, so it sees only what the source spells. It cannot
    see: an action assembled at runtime (`f"retry_{kind}"`, `"".join(...)`, a value
    read from settings or from a peer's payload); one that arrives already-formed in a
    dict this code merges (`result.update(payload)` — the service envelope's own
    `action` passes through `_service_call` untouched, and only `_public_live_fields`'
    register lookup constrains it); one written by a module outside
    `ACTION_EMITTING_MODULES`, which is a hand-kept list and the reason
    `server_auth.py` had to be added to it explicitly; and a DERIVED action that is
    wrong — a `REFUSAL_ACTIONS[...]` subscript is deliberately skipped here, and it is
    `test_every_blocking_state_is_removed_by_the_action_it_proposes` that judges
    whether the derivation is right. Runtime coverage of the register's values is what
    `test_the_register_only_names_actions_a_surface_can_offer` adds; this gate is only
    about literals somebody typed.
    """
    from ouroboros import remote_refusal_actions as registry

    vocabulary = {
        value
        for name, value in vars(registry).items()
        if name.startswith("ACTION_") and isinstance(value, str)
    }
    offenders: dict[str, list[str]] = {}
    for rel in ACTION_EMITTING_MODULES:
        path = REPO / rel
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            values: list[tuple[ast.AST, int]] = []
            if isinstance(node, ast.keyword) and node.arg == "action":
                values.append((node.value, getattr(node.value, "lineno", 0)))
            elif isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if isinstance(key, ast.Constant) and key.value == "action":
                        values.append((value, getattr(value, "lineno", 0)))
            for value, lineno in values:
                # LITERALS only. A `REFUSAL_ACTIONS[...]` subscript or a `getattr(exc,
                # "action", ...)` is a DERIVATION — the thing this whole branch is for —
                # and walking into it would flag the register's own keys as offenders.
                # A ternary or an `or` chain is flattened, because each arm is a literal
                # answer somebody chose.
                leaves = [value]
                if isinstance(value, ast.IfExp):
                    leaves = [value.body, value.orelse]
                elif isinstance(value, ast.BoolOp):
                    leaves = list(value.values)
                for leaf in leaves:
                    if not isinstance(leaf, ast.Constant) or not isinstance(leaf.value, str):
                        continue
                    text = leaf.value
                    if not text or text in ARGPARSE_ACTIONS or text in vocabulary:
                        continue
                    offenders.setdefault(text, []).append(f"{rel}:{lineno}")
    assert not offenders, (
        "action names outside `remote_refusal_actions`'s declared vocabulary: "
        f"{offenders}"
    )


# ── the action must reach the SERIALIZED projection, not only the attribute ───
#: Every projection an owner actually reads a refusal through, as
#: `(name, project) -> dict`. Each entry must expose the derived action; a new
#: projection added without one fails here rather than shipping a refusal whose
#: next step exists in memory only.
def _serialized_projections(err):
    from ouroboros.gateway.connections import _public_live_fields
    from ouroboros.remote_worker_proxy import error_dict, reconnect_failure

    wire = error_dict(err)
    live = _public_live_fields(err, default_phase="connect")
    reconnect = reconnect_failure("conn-1", err)
    return {
        "RemoteWorkspaceError.diagnostic().details": err.diagnostic().details,
        "remote_worker_proxy.error_dict()['details']": wire["details"],
        "remote_worker_proxy.error_dict()": wire,
        "remote_worker_proxy.reconnect_failure()": reconnect,
        "gateway/connections._public_live_fields()": live,
        "gateway/connections diagnostic.details": live["diagnostic"]["details"],
    }


def test_a_derived_action_reaches_every_serialized_projection():
    """A derived value that stops at the attribute never reaches the owner.

    `RemoteWorkspaceError.__init__` derives the action from `details["action"]` or
    from the register, and `diagnostic()` used to serialize `dict(self.details)`
    UNCHANGED — so a refusal that NAMED its action carried it and every one of the
    ~40 that derive theirs from the code carried nothing. The attribute was right;
    the projection was empty; and `details` is by that method's own comment the one
    slot that reaches the browser and `--json`.

    So the assertion is on the SERIALIZED output of every projection, for every code
    in the register, with NO details passed — i.e. exactly the raise sites the
    derivation was written for. Reading the constructor cannot answer this question;
    only serializing can.
    """
    from ouroboros.remote_refusal_actions import REFUSAL_ACTIONS
    from ouroboros.workspace_diagnostics import RemoteWorkspaceError

    for code, expected in sorted(REFUSAL_ACTIONS.items()):
        err = RemoteWorkspaceError(code, f"refused: {code}", phase="connect")
        assert err.action == expected, code
        for name, payload in _serialized_projections(err).items():
            assert payload.get("action") == expected, (
                f"{name} lost {code}'s derived action "
                f"(got {payload.get('action')!r}, want {expected!r})"
            )


def test_a_raiser_that_names_its_own_action_still_wins_in_every_projection():
    """Serializing the derived action must not overwrite an explicit one."""

    from ouroboros.workspace_diagnostics import RemoteWorkspaceError

    err = RemoteWorkspaceError(
        # A code whose register answer is `retry`, overridden at the raise site.
        "broker_overloaded",
        "refused",
        phase="stream",
        details={"action": "bootstrap_connection", "hint": "kept"},
    )
    for name, payload in _serialized_projections(err).items():
        assert payload.get("action") == "bootstrap_connection", name
    # And the raiser's OWN details survive alongside it, unedited.
    assert err.diagnostic().details["hint"] == "kept"
    # The derivation belongs to the PROJECTION: `self.details` stays the raiser's own
    # words, which the durable journal and the private receipt record verbatim. So a
    # refusal that passed NO details still HAS none, while its projection has one.
    derived = RemoteWorkspaceError("host_identity_changed", "refused", phase="connect")
    assert derived.details == {}
    assert derived.diagnostic().details == {"action": "retrust_host"}


def test_the_execd_half_derives_and_serializes_the_same_action():
    """One register, both sides of the wire.

    `ExecdError` is the TARGET-side twin of `RemoteWorkspaceError` and derived no
    action at all, so an execd-origin refusal reached the owner with a next step only
    because `gateway/connections` re-derived one from the code AFTER arrival — and
    the execd `--json` projection of the very same code had none.
    """
    from ouroboros.execd_state import ExecdError
    from ouroboros.remote_refusal_actions import REFUSAL_ACTIONS

    for code, expected in sorted(REFUSAL_ACTIONS.items()):
        err = ExecdError(code, f"refused: {code}", phase="execute")
        assert err.action == expected, code
        assert err.diagnostic()["details"].get("action") == expected, (
            f"the execd wire projection lost {code}'s action"
        )
    named = ExecdError(
        "broker_overloaded", "refused", phase="execute",
        details={"action": "bootstrap_connection"},
    )
    assert named.diagnostic()["details"]["action"] == "bootstrap_connection"


def test_both_diagnostic_parsers_keep_errno():
    """A parser that drops a field erases it as surely as a producer that never set it.

    The Home parser (`remote_worker_proxy.diagnostic_from_dict`) read `errno`; the
    target parser (`execd._diagnostic`) did not, so a diagnostic that round-tripped
    through the target lost the one field that says WHICH filesystem refusal it was.
    """
    from ouroboros.execd import _diagnostic
    from ouroboros.remote_worker_proxy import diagnostic_from_dict

    raw = {
        "domain": "filesystem", "code": "not_found", "message": "gone",
        "phase": "execute", "completion": "not_started", "errno": 2, "details": {},
    }
    assert diagnostic_from_dict(raw).errno == 2
    assert _diagnostic(raw).errno == 2, "the target parser dropped errno"
    # A non-integer errno is absent on BOTH sides rather than coerced.
    assert _diagnostic({**raw, "errno": "2"}).errno is None
    assert diagnostic_from_dict({**raw, "errno": "2"}).errno is None


def test_a_refusal_reaching_the_model_names_its_action_in_both_phases():
    """PREPARE printed the action and EXECUTE did not, for the SAME exception type.

    `workspace_executor.SshExecutorUnavailableError`'s own docstring says the action
    exists because it "reaches the MODEL as prose", and `tools/dispatch_prepare`
    printed it — while `tools/dispatch_execute` and `tools/verify` dropped it. One
    condition, three surfaces, two answers.

    BOUNDARY — this reads SOURCE, so it sees only the literal `[action: ` marker in
    these three files. It cannot see: an arm that builds the same prose indirectly
    (`f"[{label}: {exc.action}]"`, `"".join([...])`, a format string held in a
    constant); an arm that prints the action but reaches the model through a different
    field than the one this counts; a FOURTH module that grows an
    `except RemoteWorkspaceError` arm and drops the action, since the file list here is
    hand-kept; and whether the action printed is the RIGHT one — that is
    `test_a_derived_action_reaches_every_serialized_projection` and
    `test_every_blocking_state_is_removed_by_the_action_it_proposes`, which judge
    values rather than syntax. The RUNTIME half of this same claim is those two tests;
    this gate only asserts that the three model-facing surfaces still say something.
    """
    import ast as _ast

    for rel, arms in (
        ("ouroboros/tools/dispatch_execute.py", 2),
        ("ouroboros/tools/verify.py", 2),
        ("ouroboros/tools/dispatch_prepare.py", 2),
    ):
        source = (REPO / rel).read_text(encoding="utf-8")
        _ast.parse(source)  # the file still parses, so the count below is over real code
        assert source.count("[action: ") >= arms, (
            f"{rel} stopped naming the refusal's action on {arms} model-facing arms"
        )


def test_a_live_status_outside_the_closed_vocabulary_is_clamped(tmp_path):
    """A sixth status word is not additive but INVISIBLE, so no branch may emit one.

    `_public_live_fields` has two branches. The exception branch hardcodes `degraded`;
    the MAPPING branch passed a producer's `status` through untouched, and those
    mappings come from the broker, the transport's own `health()` and the service
    envelope — several producers, one closed five-word contract, and no boundary
    enforcing it. `_record_runtime_health` silently drops a status it does not know and
    `remote_task_state.js` reads one as *no typed status at all*, falling back to
    derivation and withdrawing the Reconnect button at the one moment a reconnect is
    the fix. `unknown` is the contract's own word for "no typed statement".
    """
    from ouroboros.gateway.connections import _public_live_fields
    from ouroboros.gateway.contracts import CONNECTION_STATUSES

    del tmp_path
    for word in ("error", "READY!", "reconnecting", "", "  "):
        row = _public_live_fields({"status": word, "phase": "connect"})
        assert row.get("status", "unknown") in CONNECTION_STATUSES, word
        if word.strip():
            assert row["status"] == "unknown", word
    # Every contract word survives verbatim, including its own casing rules.
    for word in sorted(CONNECTION_STATUSES):
        assert _public_live_fields({"status": word})["status"] == word
    # And the exception branch was already closed; it stays that way.
    from ouroboros.workspace_diagnostics import RemoteWorkspaceError

    assert _public_live_fields(
        RemoteWorkspaceError("host_identity_changed", "x", phase="connect"),
        default_phase="connect",
    )["status"] in CONNECTION_STATUSES
