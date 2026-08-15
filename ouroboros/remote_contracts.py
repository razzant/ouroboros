"""The REGISTER of Home↔execd contracts, and the one version that admits a session.

Home and execd are two independently updated artifacts. Home is the running
application; execd is a frozen bundle built from this same tree, shipped in
`assets/execd`, and installed on the target under `releases/<build>/<sha>`. They
share MORE THAN ONE contract — the wire framing, the native operation set, the
capability manifest, the export-policy document, the prepared-call shape, the
attachment channel, the durable state/journal schemas, the lease, the execution
envelope, the reconciliation ledger — and every one of them fails CLOSED on a
member it does not recognize, which is the correct behaviour and must not change.

The defect that produced this module was not any single contract. It was that a
build pair whose contracts disagree was allowed to open a session at all. Both
halves already knew each other's identity at the handshake — Home selects a
release and attests its `release_id`, execd echoes it back — and nothing compared
anything, so a Home that had added ONE rule field to the export-policy document
met a target that had never heard of it and the disagreement surfaced far away
from its cause: at PREPARE, inside an unrelated tool call, as a bare
`ValueError: export policy has unknown fields: ['marker_scoped_suffixes']`.

So there are two things here and only two.

**One version, checked at admission.** `CONTRACT_SET_VERSION` is the single
carrier of "may these two builds cooperate". It moves when any contract in
`CONTRACTS` gains, loses or reshapes a member — NOT when Home releases, because
most Home releases touch none of them, and not per contract, because a session
either understands the whole set or must not be opened. It is deliberately NOT a
new wire field: `remote_protocol.PROTOCOL_MINOR` IS this number (that module
imports it), which means it already travels in the session PREAMBLE and in both
handshake frames of every build ever shipped. That is what makes the refusal
work against a target that was installed before this module existed: it announces
contract set 0 in its preamble without being asked, and Home refuses before it
writes a single frame. A brand-new field would have been invisible on exactly the
builds the check exists to catch.

The wire keeps its own, looser question. `protocol_compatible` asks "can I PARSE
this peer?" and tolerates an older minor, because an older peer's frames really
are readable — that tolerance is what lets us read the old preamble and say
something exact about it. `contract_set_compatible` asks "may we WORK together?"
and requires equality. Two predicates over one number, each answering its own
question; there is no second version to keep in step.

**One typed refusal, wherever a contract is not understood.** Strictness is
unchanged — a policy rule this build cannot evaluate is still a policy it cannot
claim to have applied, and an unknown control kind still fails the session. What
changes is that the refusal SAYS which contract, which member, what this build
understands, and what the owner should do (`action`). `ContractDriftError`
subclasses `ValueError` on purpose: every existing caller that guards a contract
boundary with `except ValueError` keeps working, and the ones that render a
diagnostic gain fields instead of a class name.

Stdlib only, no imports from either half, so both halves may cite it — the same
property that lets `export_policy_contract` travel.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, NoReturn

# ── the contract set version ─────────────────────────────────────────────────
# 1 — the first version to exist. Its content is the contract set as of the
#     Home build that introduced this module, INCLUDING the export-policy
#     document's `marker_scoped_suffixes` rule field. Every build shipped before
#     it announces 0 (its `PROTOCOL_MINOR`), which is precisely the pair this
#     number exists to refuse.
#
# Bumping it is a deliberate edit HERE, in the same commit as the contract change
# that requires it. The test suite pins the coupling to `PROTOCOL_MINOR` so the
# number cannot be raised in one place and forgotten in the other.
CONTRACT_SET_VERSION = 1

# ── the register ─────────────────────────────────────────────────────────────
# Every contract whose shape both halves must agree on. This table is not
# decoration: `CONTRACT_SET_VERSION` means "all of these, as of this value", so a
# contract missing from it is a contract nobody promised to keep in step. It also
# supplies the owner-facing name a drift refusal reports, which is why the values
# are sentences rather than module paths.
CONTRACTS: dict[str, str] = {
    "wire_protocol": "the framed control/bulk channel and its closed control shapes",
    "native_operations": "the exact allowlist of operations execd will run",
    "capability_manifest": "the manifest Home uploads and execd admits itself against",
    "export_policy": "the export-policy rule document applied at the source",
    "import_channel": "the closed registry of blob kinds that may cross the boundary",
    "prepared_call": "the prepared-call object and the hash that binds it",
    "attachment_stage": "the task-attachment staging envelope",
    "execd_state": "execd's durable journal, spool, custody and quota schemas",
    "lease": "the generation/task lease and its answer marker",
    "execution_envelope": "the execution envelope and its typed diagnostics",
    "reconciliation": "the operation ledger a reopened session reconciles",
}

# ── owner actions ────────────────────────────────────────────────────────────
# A refusal without an action is a dead end. These are the only two an owner can
# take, and they are NOT interchangeable: one re-installs the target from the
# bundle this Home ships, the other says the bundle itself is the stale artifact
# and no amount of bootstrapping will move it.
ACTION_BOOTSTRAP = "bootstrap_connection"
ACTION_REBUILD_BUNDLE = "rebuild_execd_bundle"

CODE_UNKNOWN_MEMBER = "remote_contract_unknown_member"
CODE_EXECD_OUTDATED = "remote_execd_outdated"
CODE_EXECD_INCOMPATIBLE = "remote_execd_incompatible"
CODE_BUNDLE_OUTDATED = "execd_bundle_contract_outdated"

# A drift refusal names what it did not understand. The list is bounded because it
# rides in a diagnostic that reaches a wire trace and the owner's screen, and an
# unbounded field there is a size a peer chooses for a Home record.
MAX_REPORTED_MEMBERS = 32


class ContractDriftError(ValueError):
    """A contract member one half declared and the other has never heard of.

    Carries the four facts a reader needs and a bare ``ValueError`` never had:
    WHICH contract, WHICH members were not understood, what this build DOES
    understand, and what to do about it. ``ValueError`` remains the base so every
    boundary that already guards a normalization with ``except ValueError``
    continues to catch it unchanged.
    """

    code = CODE_UNKNOWN_MEMBER

    def __init__(
        self,
        contract: str,
        *,
        unknown: Iterable[Any],
        understood: Iterable[Any] = (),
        member: str = "fields",
        action: str = ACTION_BOOTSTRAP,
    ) -> None:
        self.contract = str(contract)
        self.member = str(member)
        self.action = str(action)
        self.unknown = _bounded_names(unknown)
        self.understood = _bounded_names(understood)
        self.contract_set_version = CONTRACT_SET_VERSION
        described = CONTRACTS.get(self.contract, self.contract)
        super().__init__(
            f"{self.contract} has unknown {self.member}: {self.unknown}; "
            f"this build understands contract set {CONTRACT_SET_VERSION} "
            f"({described}) — the peer was built from a different one"
        )

    def details(self) -> dict[str, Any]:
        """The diagnostic ``details`` block, ready to hand to a typed envelope."""

        return {
            "contract": self.contract,
            "contract_member": self.member,
            "unknown": list(self.unknown),
            "understood": list(self.understood),
            "contract_set_version": self.contract_set_version,
            "action": self.action,
        }


def refuse_unknown_members(
    contract: str,
    *,
    unknown: Iterable[Any],
    understood: Iterable[Any] = (),
    member: str = "fields",
    error_type: type[ContractDriftError] = ContractDriftError,
) -> NoReturn:
    """Fail closed on unrecognized contract members, typed and actionable.

    The ONE way a contract validator refuses something it does not understand, so
    every contract in the register answers with the same shape instead of each one
    inventing a message. It raises rather than returning: the caller's next line
    would otherwise be applying a policy it just admitted it cannot read.

    ``error_type`` exists for the contracts whose refusal must ALSO be catchable as
    the failure their own layer already handles — the wire's drift is a framing
    failure too, and a transport that stopped catching it as one would tear the
    session down through a different path than every other malformed frame. The
    subclass carries the same fields; it does not get a second message shape.
    """

    raise error_type(
        contract,
        unknown=unknown,
        understood=understood,
        member=member,
    )


def contract_set_compatible(peer_version: Any) -> bool:
    """Whether a peer announcing ``peer_version`` may share a session with us.

    EQUALITY, not a window. A contract set is all-or-nothing: the older half of an
    unequal pair cannot evaluate a rule it has never seen, and the newer half
    cannot promise the older one applied it. Compare with
    ``remote_protocol.protocol_compatible``, which answers the different and
    looser question of whether the peer's frames can be PARSED at all.
    """

    return (
        isinstance(peer_version, int)
        and not isinstance(peer_version, bool)
        and peer_version == CONTRACT_SET_VERSION
    )


def contract_skew_refusal(
    peer_version: Any,
    *,
    peer_build: str = "",
    local_build: str = "",
    action: str = ACTION_BOOTSTRAP,
    extra: Mapping[str, Any] | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """The ``(code, message, details)`` of one refused build pair.

    Built HERE, not at each admission seam, because FOUR seams can detect this one
    condition — bundle selection, the session preamble, Home's handshake response
    and execd's handshake request — and an owner told "outdated" in one place and
    "incompatible" in another has to work out that they are the same fact. The
    wording is deliberately perspective-neutral, because two of those seams run on
    Home and one runs on the target.

    The two codes are diagnosis, not different actions. ``remote_execd_outdated``
    means the peer is BEHIND, ``remote_execd_incompatible`` that it is AHEAD; both
    are resolved by ``bootstrap_connection``, which installs exactly the release
    this Home ships and switches the ``current`` symlink to it in either direction.
    ``execd_bundle_contract_outdated`` is the one case bootstrap cannot resolve —
    the stale artifact is the bundle itself, so re-installing it changes nothing.
    """

    behind = (
        isinstance(peer_version, int)
        and not isinstance(peer_version, bool)
        and peer_version < CONTRACT_SET_VERSION
    )
    if action == ACTION_REBUILD_BUNDLE:
        code = CODE_BUNDLE_OUTDATED
        remedy = (
            "Bootstrap cannot resolve this: the execd bundle shipped with this "
            "Ouroboros build declares an older contract set than the build itself, "
            "so the bundle is the artifact that has to be rebuilt."
        )
    else:
        code = CODE_EXECD_OUTDATED if behind else CODE_EXECD_INCOMPATIBLE
        remedy = (
            "Bootstrap the connection: it installs the execd release this Ouroboros "
            "build ships and points the target's `current` release at it."
        )
    message = (
        "Home and execd do not share a Home↔execd contract set "
        f"(the peer announces {peer_version!r}, this build requires "
        f"{CONTRACT_SET_VERSION}). {remedy}"
    )
    details: dict[str, Any] = {
        "peer_contract_set": peer_version if isinstance(peer_version, int) else str(peer_version),
        "required_contract_set": CONTRACT_SET_VERSION,
        "peer_build": str(peer_build or "unknown"),
        "local_build": str(local_build or "unknown"),
        "action": action,
        "contracts": sorted(CONTRACTS),
    }
    details.update(dict(extra or {}))
    return code, message, details


def _bounded_names(values: Iterable[Any]) -> list[str]:
    names = sorted({str(value) for value in values})
    return names[:MAX_REPORTED_MEMBERS]
