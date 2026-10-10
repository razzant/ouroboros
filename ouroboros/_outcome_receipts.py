"""Private receipt parsing and reconciliation helpers for typed outcomes.

Verification receipts are written only by ``tools/verify.py``; shell/process tools write
none. Receipts reconcile by one typed identity key (kind + value), never across kinds.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ouroboros.shell_parse import canonical_command_text
from ouroboros.utils import truncate_review_artifact

# How many observed paths one reviewer projection carries inline. The bound stays;
# what may never happen is bounding SILENTLY — `receipt_identity_projection` always
# emits the omitted COUNT and a durable hash of the FULL normalized path-set identity.
IDENTITY_PATH_LIMIT = 20

# A failed verification receipt (``status=="fail"``) is "reconciled" ONLY by a LATER
# genuine grounding receipt — a passing run-kind check (``pass``) or an observed artifact
# (``observed``) — OF THE SAME VERIFICATION (v6.78.0, owner Q28=B: the same typed identity
# KEY; an identity-LESS red keeps the older any-later-green rule; see `_reconciles`). A
# later ``declared`` does NOT reconcile a red: that would let an agent see red, then
# declare-away the finalization nudge. So this is deliberately NARROWER than
# ``outcomes._RECEIPT_GROUNDING_STATUSES`` (no ``declared``). Lives HERE, next to the
# reconciliation core, so the outstanding-state helpers have a default and `outcomes.py`
# keeps re-exporting it under its historical name.
RED_RECONCILING_STATUSES = frozenset({"pass", "observed"})

# The identity KIND of a receipt that names no verification at all. Kept as a name because
# "has no identity" is a case every relation below must handle explicitly.
IDENTITY_KIND_NONE = "none"

# How a receipt's ``check`` text was RENDERED from the argv that ran — a version stamp on
# the stored string, because the renderer changed and the string alone cannot say which
# one produced it (round 8).
#
# v6.78.0 switched ``tools/verify.py`` from ``" ".join(argv)`` to ``shlex.join(argv)``
# because a space-join is not injective. The cost was argued as safe in one direction —
# an old and a new receipt for the SAME argv render differently, fail to reconcile, and a
# non-reconciling red stays red — but the OTHER direction was never checked: an old red
# from argv ``["echo", "a b"]`` and a new green from argv ``["echo", "a", "b"]`` both read
# ``echo a b``, so the green cleared the red. A false green, from the very change made to
# remove one. The stamp closes it: the rendering is part of the check identity, so two
# receipts rendered by different writers are never the same verification.
#
# Any unrecognised value is its own namespace rather than an error — a future renderer is
# automatically incomparable with today's, which is the safe answer without a code change.
CHECK_RENDERING_SHLEX_JOIN = "shlex_join"
# The absent stamp: written before v6.78.0, by ``" ".join(argv)``, and therefore AMBIGUOUS
# — the text cannot be re-tokenized back to the argv that ran. Not "equal to nothing":
# two legacy receipts still match each other, which is exactly the behaviour they had
# before the upgrade and the most that can be recovered from data already on disk. What
# they may not do is match a versioned receipt: that pairing is UNKNOWN, and unknown must
# never clear a red.
CHECK_RENDERING_UNVERSIONED = "unversioned"
# The `declared` contract kind stores the agent's OWN check text verbatim — never a render
# of an executed argv — so it is a third rendering, not either of the two above. It reaches
# no reconciliation today (a `declared` receipt is neither a red candidate nor a grounding),
# but stamping it keeps the disclosure true rather than labelling agent prose as a renderer
# output, and keeps it from ever silently sharing a key with a command that actually ran.
CHECK_RENDERING_DECLARED_TEXT = "declared_text"


@dataclass(frozen=True)
class ReviewRunSelection:
    """Current review authority plus retained superseded audit evidence."""

    all_runs: List[Dict[str, Any]]
    current_runs: List[Dict[str, Any]]
    superseded_only_acceptance_gap: bool
    superseded_aggregate_signals: List[str]
    current_candidate_unaccepted: bool
    has_replacement: bool


def select_current_review_runs(
    review_runs: Any,
    *,
    delivery_candidate: Any,
    review_decision: Any,
) -> ReviewRunSelection:
    """Select current review runs without erasing conservative stale failures."""
    all_runs = [
        run for run in (review_runs or [])
        if isinstance(run, dict) and run.get("authority") != "agent_advisory"
    ]
    current_runs = [run for run in all_runs if not run.get("superseded_by_revision")]
    candidate = delivery_candidate if isinstance(delivery_candidate, dict) else {}
    binding = candidate.get("acceptance_binding")
    binding = binding if isinstance(binding, dict) else {}
    decision = review_decision if isinstance(review_decision, dict) else {}
    candidate_unaccepted = bool(candidate) and binding.get("authoritative") is False
    superseded_signals = [
        str(run.get("aggregate_signal") or "").upper() for run in all_runs
    ]
    acceptance_gap = bool(
        all_runs
        and not current_runs
        and candidate_unaccepted
        and not (decision.get("panel_id") and decision.get("binding_hash"))
        and "FAIL" not in superseded_signals
        and "DEGRADED" not in superseded_signals
    )
    # A host decision that names a panel by id AND binding applied THAT panel's
    # verdict (an earlier revision accepted on the reviewers' word): when no run
    # is current, the named run speaks alone and an older superseded FAIL stays
    # audit evidence instead of outvoting the applied PASS. With no named panel
    # the conservative rule stands: every stale run is retained.
    named = [
        run for run in all_runs
        if decision.get("panel_id") and decision.get("binding_hash")
        and str(run.get("panel_id") or "") == str(decision.get("panel_id"))
        and str(run.get("binding_hash") or "") == str(decision.get("binding_hash"))
    ]
    selected = [] if acceptance_gap else (current_runs if current_runs else (named or all_runs))
    return ReviewRunSelection(
        all_runs=all_runs,
        current_runs=selected,
        superseded_only_acceptance_gap=acceptance_gap,
        superseded_aggregate_signals=superseded_signals,
        current_candidate_unaccepted=candidate_unaccepted,
        # A decision-named panel replaces the stale runs for the ledger as well:
        # an older FAIL reads as superseded, not as live failure evidence.
        has_replacement=bool(current_runs) or bool(named),
    )


def review_run_ledger_status(
    run: Dict[str, Any],
    selection: ReviewRunSelection,
) -> tuple[str, bool]:
    """Project one acceptance run into the verification ledger."""
    signal = str(run.get("aggregate_signal") or "").upper()
    superseded = bool(run.get("superseded_by_revision")) and (
        selection.has_replacement
        or (selection.current_candidate_unaccepted and signal == "PASS")
    )
    failed = run.get("aggregate_signal") in {"FAIL", "DEGRADED"} or bool(run.get("degraded"))
    if failed and not superseded and review_runs_only_awaited([run]):
        return "not_evaluated", False  # reviewers had not answered yet: a gap, never a failed verification
    return ("superseded" if superseded else ("failed" if failed else "ok")), superseded


def plan_review_awaiting(*records: Any) -> bool:
    """Whether a task record carries the typed fact of a clean finish over a plan review
    that was only awaited (``outcome_axes.execution.plan_review``)."""
    from ouroboros.review_projection import AWAITING_PROJECTION

    for record in records:
        axes = record.get("outcome_axes") if isinstance(record, dict) else None
        execution = axes.get("execution") if isinstance(axes, dict) else None
        if isinstance(execution, dict) and execution.get("plan_review") == AWAITING_PROJECTION:
            return True
    return False


def review_runs_only_awaited(runs: List[Dict[str, Any]]) -> bool:
    """Whether the runs lack a verdict ONLY because reviewers had not answered yet.

    Typed rows alone: every run without a PASS holds at least one slot released at the
    dispatch barrier that carries no answer, and nothing else beside it — every other
    slot answered a PASS whose tier, if any, is ``solved``. A recorded FAIL, a PASS that
    judged the work blocked or best-effort, or a failed, refused, unresolved or
    parse-degraded slot, is a real outcome and keeps its own word; an answer that has
    not arrived is a gap, never a degradation of the task."""
    from ouroboros.review_records import review_slot_awaiting

    unsettled = [run for run in runs
                 if str(run.get("aggregate_signal") or "").upper() != "PASS" or run.get("degraded")]
    for run in unsettled:
        awaited = 0
        for row in run.get("actors") or []:
            if not isinstance(row, dict):
                return False
            parsed = row.get("parsed")
            if review_slot_awaiting(row):
                if row.get("status") == "ok" or parsed is not None or str(row.get("raw_text") or "").strip():
                    return False  # a pending row that carries an answer is judged by its answer
                awaited += 1
            elif (row.get("status") != "ok" or str(row.get("signal") or "").upper() != "PASS"
                  or (isinstance(parsed, dict) and str(parsed.get("outcome_tier") or "solved") != "solved")):
                return False
        if not awaited or str(run.get("aggregate_signal") or "").upper() == "FAIL":
            return False
    return bool(unsettled)


def canonical_path_set(paths: Any) -> tuple[str, ...]:
    """The ONE canonical form of a receipt path SET: de-duplicated and sorted, and
    otherwise VERBATIM.

    No whitespace is touched, neither interior nor edge. A leading or trailing space is
    a legal filename byte, so ``"a.md "`` and ``"a.md"`` name two different files and
    trimming them together let a green observation of one close a red observation of the
    other (round 5) — the same defect ``" ".join(text.split())`` caused for INTERIOR runs
    of whitespace one round earlier. A normalization that discards a byte the identity
    depends on is not a normalization; only the empty string is dropped, because it names
    nothing.

    De-duplication and sorting happen HERE, on the RAW values, so everything derived
    downstream — the compared identity, the hash, the carried items and the omitted
    count — describes the very same set. The order is canonicalize → render → bound,
    never render → dedup: rendering is lossy (redaction, truncation), so deduplicating
    after it silently drops distinct values while the omitted count still says zero.
    """
    values = paths if isinstance(paths, (list, tuple)) else []
    return tuple(sorted({text for text in (str(path) for path in values) if text}))


@dataclass(frozen=True)
class ReceiptIdentity:
    """The canonical identity of ONE verification receipt — derived ONCE, then used for
    comparison, hashing, counting and projection alike.

    The three components are INDEPENDENT and each empty when the receipt does not carry
    it: the reviewer/task-authored ``criterion_id``; the CANONICAL ``check`` command
    text (structural — see ``shell_parse.canonical_command_text``); the canonical
    observed ``paths`` set of the command-less artifact-observation class. Every
    consumer in this module reads THIS object, so the "identity" the reconciliation
    compares, the one the hash covers and the one the reviewer projection carries can
    no longer be three subtly different notions (round-4 review: they were, and the
    comparison form was lossy where the projection form was not).

    Sameness, however, is decided by ``key`` alone — never by the components one after
    another. See ``key``.
    """

    criterion_id: str
    check: str
    paths: tuple[str, ...]
    check_rendering: str = CHECK_RENDERING_UNVERSIONED

    @property
    def key(self) -> tuple[str, str]:
        """THE identity of this receipt: ONE typed ``(kind, value)`` key. Two receipts
        name the same verification when the kind AND the value match, and never across
        kinds.

        The kind is the most specific thing the receipt actually carries — the
        task/reviewer-authored ``criterion_id``, else the canonical ``check`` command
        text, else the observed ``paths`` set of the command-less artifact-observation
        class. That ordering picks the key; it is not a matching rule, and no second
        component is ever consulted once the key is chosen.

        Why a single key (round-5 CRITICAL): the identity used to be a FALLBACK CHAIN —
        match on ``criterion_id``, else on ``check``, else on ``paths``. A chain is not
        an equivalence relation. It is not transitive: ``{c1, check}`` matched
        ``{check}``, ``{check}`` matched ``{c2, check}``, yet ``c1`` and ``c2`` are
        explicitly different criteria. So one later ``check``-only pass reconciled BOTH
        reds, "collapse the candidates onto the identity they name" was not even
        well defined, and the outstanding set came out order-dependent. Keying makes the
        relation the KERNEL of a function — reflexive, symmetric and transitive by
        construction — and makes an existing ``criterion_id`` authoritative
        STRUCTURALLY: a ``check``-only pass simply has a different key from a
        criterion-keyed red and cannot touch it.

        It is a SIMPLIFICATION (strictly fewer branches than the chain) and it fails in
        the SAFE direction: strictly fewer reconciliations, so a red that the chain used
        to clear can now stay open. In particular a re-run that OMITS the
        ``criterion_id`` it carried before no longer clears its own red — the loss is one
        advisory nudge and one advisory reviewer flag, never a false green. If that
        omission tolerance is ever genuinely needed, the ONLY sound route is to carry the
        id forward STRUCTURALLY at receipt ingress (``tools/verify.py``), never to infer
        it back from shared command text.
        """
        for kind, component, _ in IDENTITY_KINDS:
            value = component(self)
            if value:
                return (kind, value)
        return (IDENTITY_KIND_NONE, "")

    @property
    def criterion_key(self) -> tuple[str, str]:
        """The typed key of the MASKED path, whose only usable identity is the
        ``criterion_id`` (``_reconciles_masked`` explains why the check text cannot
        serve). Same shape and same rules as ``key`` — one kind, one value."""
        return ("criterion_id", self.criterion_id) if self.criterion_id else (IDENTITY_KIND_NONE, "")

    @property
    def check_identity_source(self) -> str:
        """The INJECTIVE serialization of the check identity — the canonical command text
        PAIRED with the rendering that produced the stored string, empty when there is no
        check. Same shape and same reason as ``paths_identity_source``: the thing being
        identified is not the text alone. Two receipts written by different renderers can
        spell the same text while having run different argvs, so the pair is the identity
        and the text by itself is not (round 8)."""
        return (
            json.dumps([self.check_rendering, self.check], ensure_ascii=False)
            if self.check else ""
        )

    @property
    def paths_text(self) -> str:
        """The DISCLOSED text rendering of the path set (one path per line)."""
        return "\n".join(self.paths)

    @property
    def paths_identity_source(self) -> str:
        """The INJECTIVE serialization of the path set that ``paths_identity_sha256``
        hashes — empty when there is no path set, so a receipt without one claims no
        hash. JSON rather than ``paths_text``: a path may legally contain a newline,
        and a line-joined rendering would hash ``["a\\nb"]`` and ``["a", "b"]``
        identically — a hash that cannot separate two different sets is not an
        identity."""
        return json.dumps(list(self.paths), ensure_ascii=False) if self.paths else ""


# THE closed set of reconciliation identity KINDS, in precedence order (most specific
# first) — one row per kind, and the row carries everything that is true OF that kind:
# its name, how to read its value off a `ReceiptIdentity`, and whether its identity is
# CANONICAL COMMAND TEXT whose token spacing normalizes.
#
# That third column is not decoration. Rounds 6 and 7 were the same defect twice: a
# disclosure claiming "this receipt reconciles on whitespace-normalized command text"
# for a kind that normalizes nothing (an id-less masked pass, then an `artifact_paths`
# observation whose set is compared byte-for-byte — `canonical_path_set` deliberately
# preserves every whitespace byte of a filename). Answering per kind ONCE, here, makes
# the answer TOTAL: `receipt_expected_whitespace_normalized` looks the kind up rather
# than testing for the kinds it happens to remember, and a FOURTH kind added to this
# table must state its own answer — it cannot inherit a default, because there is no
# default to inherit. The `paths` value is the INJECTIVE serialization, not the
# line-joined rendering: a path may legally contain a newline, and a key that cannot
# separate two different sets is not an identity.
IDENTITY_KINDS: tuple[tuple[str, Callable[["ReceiptIdentity"], str], bool], ...] = (
    ("criterion_id", lambda identity: identity.criterion_id, False),
    ("check", lambda identity: identity.check_identity_source, True),
    ("artifact_paths", lambda identity: identity.paths_identity_source, False),
)

# Kind → "is this kind's identity canonical command text?", total over every kind a key
# can carry, INCLUDING the no-identity kind. A `KeyError` here is the gate working: a new
# kind that skipped the table fails loudly instead of defaulting to a false claim.
KIND_NORMALIZES_COMMAND_TEXT: Dict[str, bool] = {
    **{kind: normalizes for kind, _, normalizes in IDENTITY_KINDS},
    IDENTITY_KIND_NONE: False,
}


def receipt_canonical_identity(receipt: Dict[str, Any]) -> ReceiptIdentity:
    """THE identity derivation for verification receipts — the single seam every
    comparison, hash, count and projection in this module goes through."""
    return ReceiptIdentity(
        criterion_id=str(receipt.get("criterion_id") or "").strip(),
        check=canonical_command_text(receipt.get("check")),
        paths=canonical_path_set(receipt.get("paths")),
        check_rendering=receipt_check_rendering(receipt),
    )


def receipt_check_rendering(receipt: Dict[str, Any]) -> str:
    """Which renderer produced this receipt's stored ``check`` text — the declared stamp,
    or ``unversioned`` when the receipt predates the stamp (see the constants above)."""
    return str(receipt.get("check_rendering") or "").strip() or CHECK_RENDERING_UNVERSIONED


def receipt_identity_parts(receipt: Dict[str, Any]) -> tuple[str, str, str]:
    """The three identity components of one verification receipt as TEXT — a DISCLOSURE
    of ``receipt_canonical_identity``, kept because the ledger and the architecture docs
    name it.

    A disclosure, NOT the comparison. Reconciliation selects exactly ONE typed key from
    these components in precedence order and compares that key alone
    (``ReceiptIdentity.key``); it never matches component by component and never falls
    back from one component to another. This docstring used to say the opposite —
    "matches these COMPONENT-wise" — which is the pre-v6.78.0 fallback chain whose removal
    is the central safety property of the change, so the sentence described the one
    behaviour that must not be reconstructed from it (round 8)."""
    identity = receipt_canonical_identity(receipt)
    return (identity.criterion_id, identity.check, identity.paths_text)


def receipt_identity(receipt: Dict[str, Any]) -> tuple[str, str]:
    """THE typed reconciliation key of ONE verification receipt (v6.78.0, owner Q28=B) —
    ``ReceiptIdentity.key``, named here because the ledger, the acceptance summary and
    the architecture docs all call it "the receipt identity".

    Kind ``none`` with an empty value means the receipt names no verification at all and
    can therefore reconcile nothing. The artifact-observation class runs no command, so
    its observed path set is the only identity it has — without it a red "report.md is
    missing" could never be cleared by the byte-identical green observation after the
    file was written. This is content-ADDRESSING, not content-derived lookup of a
    host-minted row (see the DEVELOPMENT.md anti-pattern section).
    """
    return receipt_canonical_identity(receipt).key


def receipt_is_masked_pass(receipt: Dict[str, Any]) -> bool:
    """Whether this receipt is a MASKED pass — a green whose check pipeline can launder
    the real exit code (``tools/verify._check_has_exit_masking``). THE predicate that
    selects the masked reconciliation MODE; also the outstanding-masked candidate test,
    so "which receipts the masked rule governs" is decided in exactly one place."""
    return str(receipt.get("status") or "") == "pass" and bool(receipt.get("check_exit_masking"))


def receipt_reconciliation_key(receipt: Dict[str, Any], *, masked: bool) -> tuple[str, str]:
    """THE key that decides whether a later grounding reconciles this receipt, in the
    reconciliation MODE that applies — the ONE projection both the comparison and the
    disclosure read.

    Two modes, because there are two rules. The RED/default mode uses the full typed
    identity (``ReceiptIdentity.key``: ``criterion_id``, else canonical ``check`` text,
    else the observed ``paths`` set). The MASKED mode uses ``criterion_key`` — the
    ``criterion_id`` alone — because ``_reconciles_masked`` cannot consult check text at
    all (the remediation the host asks for necessarily changes it).

    Round-6 CRITICAL, and the reason this function exists rather than two labels: the
    disclosure used to RE-DERIVE the authority instead of reading the deciding one, so an
    id-less masked pass — reconciled by ANY later clean grounding — was projected as
    ``reconciliation_identity="check"`` with ``expected_whitespace_normalized=true``. That
    is host-attested evidence making a false claim about its own basis, which is the exact
    defect class this surface exists to eliminate. The decider and the reporter now call
    the same function; a third rule cannot drift them apart, because there is nowhere left
    to re-derive from.
    """
    identity = receipt_canonical_identity(receipt)
    return identity.criterion_key if masked else identity.key


def receipt_disclosed_reconciliation_key(receipt: Dict[str, Any]) -> tuple[str, str]:
    """The key that governs THIS receipt on its own — the mode read off the receipt, for
    the projections that must describe one row rather than compare a pair."""
    return receipt_reconciliation_key(receipt, masked=receipt_is_masked_pass(receipt))


def receipt_expected_whitespace_normalized(receipt: Dict[str, Any]) -> bool:
    """DISCLOSED flag (owner Q28=B): the key that actually governs this receipt is the
    CANONICAL CHECK COMMAND TEXT, so only a later green whose command canonicalizes to
    the same text clears it — the same tokens however spaced, but a cosmetically different
    re-run (``pytest x.py`` vs ``pytest x.py -v``) or a changed quoted ARGUMENT does NOT.

    Derived from the key's KIND through the closed ``KIND_NORMALIZES_COMMAND_TEXT``
    table — one lookup, total over every kind, never a list of the kinds this function
    happens to remember. That is the whole fix for a defect this phase hit twice: the flag
    read True for an id-less masked pass (round 6), which reconciles without consulting
    command text at all, and then for ``artifact_paths`` (round 7), whose observed set is
    compared BYTE-FOR-BYTE because a leading or trailing space is a legal filename byte.
    Both were host-attested evidence misdescribing its own basis. So it is true for
    ``check`` and false for ``criterion_id`` (an authored name), ``artifact_paths`` (no
    normalization whatsoever) and ``none`` (no key at all) — and a fourth kind gets its
    answer from its own table row rather than from whatever this expression defaulted to.
    Advisory, never a gate."""
    kind, value = receipt_disclosed_reconciliation_key(receipt)
    return bool(value) and KIND_NORMALIZES_COMMAND_TEXT[kind]


def _reconciles(later: Dict[str, Any], earlier: Dict[str, Any]) -> bool:
    """Whether ``later`` grounds the SAME verification as ``earlier`` (Q28=B).

    ONE comparison: the two typed identity keys are equal (``ReceiptIdentity.key``,
    which explains why a single key replaced the old component-by-component fallback
    chain). Equal kind AND equal value, never a match across kinds — so an existing
    ``criterion_id`` is authoritative structurally: a ``check``-only green has a
    ``check`` key and cannot reach a criterion-keyed red at all.

    An EARLIER receipt whose key is ``none`` falls back to the pre-v6.78.0 rule (any
    later grounding reconciles it): it names no verification, so the narrowing would
    buy nothing and only mint an UNCLEARABLE flag — one malformed early call
    (``artifact_observation`` with no ``artifact_paths`` writes a red with empty
    ``paths`` and no ``check``, verify.py) would otherwise keep ``unreconciled_red``
    true for the rest of the task no matter how many real green verifications followed.
    Restricted to keyed receipts — which is every pair the outstanding SET compares —
    the relation is exactly key equality, hence an equivalence."""
    earlier_key = receipt_reconciliation_key(earlier, masked=False)
    if earlier_key[0] == IDENTITY_KIND_NONE:
        return True
    return receipt_reconciliation_key(later, masked=False) == earlier_key


def _reconciles_masked(later: Dict[str, Any], earlier: Dict[str, Any]) -> bool:
    """Whether the CLEAN ``later`` receipt re-grounds the masked ``earlier`` one.

    Same shape as ``_reconciles`` on a narrower key: ``ReceiptIdentity.criterion_key``.
    The check text cannot serve here — a masked receipt's only text identity is its own
    MASKED command, and the sensor (``tools/verify._check_has_exit_masking``) is a pure
    function of argv, so a byte-identical re-run is re-flagged masked and could never be
    the clean reconciler; requiring text equality would make the flag unclearable by the
    very remediation the host asks for ("drop the masking pipe"), unlike the RED path
    where remediation preserves the command text.

    So an ``earlier`` masked receipt that HAS a ``criterion_id`` is re-grounded only by a
    later clean receipt naming that same id — round-5: the old form let a later clean
    receipt that merely OMITTED its id reconcile every open masked criterion at once,
    the same non-transitivity as the red path. Only an earlier receipt with NO id keeps
    the any-later-clean fallback, for the same reason as in ``_reconciles``."""
    earlier_key = receipt_reconciliation_key(earlier, masked=True)
    if earlier_key[0] == IDENTITY_KIND_NONE:
        return True
    return receipt_reconciliation_key(later, masked=True) == earlier_key


def _same_verification(one: Dict[str, Any], other: Dict[str, Any]) -> bool:
    """Whether two OUTSTANDING receipts name the SAME verification: equal typed identity
    keys, both keyed.

    Direct key equality rather than ``_reconciles`` in both directions — that relation
    answers True for ANY pair whose earlier member is identity-LESS (a fallback so one
    malformed receipt cannot mint an unclearable red), and using it as equality would
    fold every identity-less red into whichever red happened to be collected first.
    Receipts with no key are never equal, not even to each other: they are
    indistinguishable, not known-equal."""
    key = receipt_reconciliation_key(one, masked=False)
    return key[0] != IDENTITY_KIND_NONE and key == receipt_reconciliation_key(other, masked=False)


def _same_masked_verification(one: Dict[str, Any], other: Dict[str, Any]) -> bool:
    """Same-verification equality for the MASKED path — the same rule as
    ``_same_verification`` on ``criterion_key``, whose kind is ``none`` for a receipt
    carrying no ``criterion_id``, so unrelated masked checks are never collapsed."""
    key = receipt_reconciliation_key(one, masked=True)
    return key[0] != IDENTITY_KIND_NONE and key == receipt_reconciliation_key(other, masked=True)


def disclosed_list_projection(
    values: Any,
    *,
    key: str,
    limit: int,
    bound: Callable[[Any, int], str] = truncate_review_artifact,
    item_cap: int = 200,
    item: Optional[Callable[[Any], Any]] = None,
    hash_key: str = "",
    hash_source: Optional[str] = None,
) -> Dict[str, Any]:
    """The ONE shared way a cognitive-review surface carries a BOUNDED list (BIBLE P1).

    The bound stays — reviewer packets are budgeted — but it may never be SILENT. Every
    projection emits ``<key>`` (the carried items, each string bounded through the SSOT
    ``utils.truncate_review_artifact`` or the caller's redacting wrapper of it, so a
    clipped string carries its own omission note) AND ``<key>_omitted``, the EXACT number
    of items left out (0 when nothing was, stated rather than left to inference). A
    hand-rolled ``[:N]`` on one of these surfaces is the anti-pattern this replaces:
    it destroys evidence and leaves no trace that evidence was destroyed.

    ``hash_key`` additionally records a sha256 of the FULL set, for the callers whose
    complete evidence is NOT reachable from the store the bounded row lives in (the
    receipt path-set identity that ``_reconciles`` compares; the native-retrieval URL
    set, which only exists inside per-call observability payloads). Callers whose whole
    row IS durable beside the projection — every ``verification_receipt_ledger_row``
    field, since the complete receipt is appended to the per-task
    ``verification_receipts.jsonl`` — pass no ``hash_key``: the omitted count plus that
    store is already the reference, and a per-row hash would be noise.
    """
    items = list(values) if isinstance(values, (list, tuple)) else []
    render = item if item is not None else (lambda value: bound(str(value), item_cap))
    shown = items[: max(0, limit)]
    out: Dict[str, Any] = {
        key: [render(value) for value in shown],
        f"{key}_omitted": max(0, len(items) - len(shown)),
    }
    if hash_key:
        source = (
            hash_source
            if hash_source is not None
            else json.dumps(items, ensure_ascii=False, sort_keys=True, default=str)
        )
        if source:
            out[hash_key] = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return out


def _lifecycle_row(
    item: Any,
    *,
    bound: Callable[[Any, int], str] = truncate_review_artifact,
    cap: int = 200,
) -> Any:
    """One ``artifact_lifecycle`` entry with its path DISCLOSED-bounded (the structural
    existence/surface flags are kept whole; a non-dict row is bounded as text).

    ``bound`` is the caller's string bound — the SSOT truncator by default, and the
    REDACTING ``review_evidence._accept_redact_cap`` wherever the row reaches an
    external reviewer, since a lifecycle path is raw host filesystem surface."""
    if not isinstance(item, dict):
        return bound(str(item), cap)
    if "path" not in item:
        return dict(item)
    return {**item, "path": bound(str(item.get("path") or ""), cap)}


def receipt_identity_projection(
    receipt: Dict[str, Any],
    *,
    bound: Callable[[Any, int], str] = truncate_review_artifact,
    check_cap: int = 300,
    path_cap: int = 200,
    path_limit: int = IDENTITY_PATH_LIMIT,
) -> Dict[str, Any]:
    """The ONE shared disclosed projection of a receipt's reconciliation identity.

    Both fixed reviewer projections (``verification_receipt_ledger_row`` and
    ``review_evidence._accept_verification_summary``) render the identity through
    THIS function, so the components reconciliation actually compared can never
    diverge between them: every component that participates —
    ``criterion_id`` / normalized ``check`` text / observed ``paths`` set — is
    carried, whichever one ``receipt_identity`` selected as primary.

    Bounding is DISCLOSED, never silent (BIBLE P1). Long strings go through the
    caller's ``bound`` — the SSOT ``truncate_review_artifact`` by default, and the
    redacting ``_accept_redact_cap`` (which wraps the same helper) on the reviewer
    path, because a receipt's command text is raw host stdout. The path LIST is
    additionally capped in length, so it always reports ``paths_omitted`` and, when
    a path set exists, ``paths_identity_sha256`` — the hash of the FULL normalized
    set that ``_reconciles`` compares. The bounded list is therefore checkable
    against the whole identity, and the complete receipt stays durable in the
    per-task ``verification_receipts.jsonl``.
    """
    identity = receipt_canonical_identity(receipt)
    return {
        # AGENT-SUPPLIED text, exactly like the `check` beside it — so it goes through
        # the caller's `bound` on the SAME cap, never inserted raw (round 3): the
        # reviewer path's `bound` is the REDACTING `_accept_redact_cap`, and a raw
        # `criterion_id` would have been the one identity component reaching external
        # acceptance reviewers unredacted, and unbounded enough to overflow the pack's
        # immutable core. Every OTHER field here is either bounded through `bound` too
        # (`check`, `paths`) or host-minted and closed (`reconciliation_identity` is one
        # of four fixed kind strings, `expected_whitespace_normalized` a bool,
        # `paths_omitted` an int, `paths_identity_sha256` a hex digest).
        "criterion_id": bound(identity.criterion_id, check_cap),
        # The CANONICAL check — the very text the identity compares. Emitting the raw
        # `receipt["check"]` here (round 5) contradicted the single-derivation claim this
        # function exists to make: the reviewer read one spelling while reconciliation
        # used another, and the two disagree exactly where it matters (a quoted operator,
        # an argv spelling). The raw text stays durable in `verification_receipts.jsonl`.
        "check": bound(identity.check, check_cap),
        # WHICH RENDERER produced the stored `check` text (`shlex_join` | `unversioned`).
        # Host-minted and closed. Disclosed because it PARTICIPATES in the check identity:
        # two receipts can carry the same `check` string, both read
        # `reconciliation_identity="check"`, and still be different verifications when the
        # renderings differ — without this the reviewer would see an unreconciled red beside
        # a byte-identical green and read the host as broken (round 8).
        "check_rendering": receipt_check_rendering(receipt),
        # WHICH identity clears this receipt (`criterion_id` | `check` |
        # `artifact_paths` | `none`), and whether it is canonical TEXT, not an id. Read
        # off the SHARED mode-aware key `_reconciles`/`_reconciles_masked` compare, so a
        # masked pass reports the authority that really decides it — `criterion_id`, or
        # `none` for the any-later-clean fallback — and never the check text it ignores.
        "reconciliation_identity": receipt_disclosed_reconciliation_key(receipt)[0],
        "expected_whitespace_normalized": receipt_expected_whitespace_normalized(receipt),
        # The observed-path SET of the command-less artifact-observation class — that
        # class runs no command, so this IS its identity. The CANONICAL set feeds the
        # projection, so the carried items, the omitted count and the hash all describe
        # the SAME set (round 4: they did not — the list was the raw ordered one, so a
        # duplicate or a whitespace variant claimed an omission the hashed identity had
        # never contained). Bounded through the SHARED disclosed-list projection:
        # count + full-set hash, never a silent slice.
        **disclosed_list_projection(
            identity.paths,
            key="paths",
            limit=path_limit,
            bound=bound,
            item_cap=path_cap,
            hash_key="paths_identity_sha256",
            hash_source=identity.paths_identity_source,
        ),
    }


def verification_receipt_ledger_row(receipt: Dict[str, Any]) -> Dict[str, Any]:
    """FIXED ledger projection of one host-attested verify_and_record receipt.

    A new receipt key is silently dropped unless it is added HERE (or to the shared
    ``receipt_identity_projection`` this row splats in). Bounded for ledger size —
    always with a DISCLOSED marker/omitted count, never silently; the full receipt
    stays durable in the per-task receipt store."""
    return {
        "kind": "verification_receipt",
        "status": str(receipt.get("status") or "unknown"),
        "contract_kind": str(receipt.get("contract_kind") or ""),
        # v6.78.0 (Q28=B): `criterion_id` / `check` / `paths` (+ `paths_omitted` and
        # `paths_identity_sha256`) and the disclosed identity rule all come from the
        # SHARED `receipt_identity_projection`, so this row and the acceptance
        # `verification_summary` can never disagree about the identity reconciliation used.
        **receipt_identity_projection(receipt),
        "expected": truncate_review_artifact(receipt.get("expected"), limit=200),
        # Verification SEMANTICS so a reviewer sees how `expected` was matched
        # (substring-only is weak evidence for a metric-graded task).
        "expected_match": str(receipt.get("expected_match") or "substring"),
        "matched": receipt.get("matched"),
        "returncode": receipt.get("returncode"),
        # Disclosure keys (node-runtime sprint, D6/R4) — carried ONLY when the
        # receipt has them, so historical rows and non-run kinds stay
        # byte-identical. `duration_ms`: how long the check's process lived.
        # `signal`: POSIX signal name of a killed check (returncode < 0) —
        # a 9ms SIGKILL is distinguishable from an honest red. `resolved_runtime`:
        # the physical executable that ran when the interpreter resolver
        # substituted one; ABSENT means the recorded `check` argv executed as
        # written. Disclosure only — receipt identity/reconciliation (the shared
        # `receipt_identity_projection` above) reads none of these.
        **({"duration_ms": receipt.get("duration_ms")} if receipt.get("duration_ms") is not None else {}),
        **({"signal": str(receipt.get("signal") or "")} if receipt.get("signal") else {}),
        **({"resolved_runtime": truncate_review_artifact(receipt.get("resolved_runtime"), limit=300)} if receipt.get("resolved_runtime") else {}),
        "summary": truncate_review_artifact(receipt.get("summary"), limit=300),
        # C: after-only artifact-lifecycle flag (a check that built then deleted a
        # declared deliverable). Bounded through the SHARED disclosed-list projection —
        # each list reports its own `_omitted` count, and the COMPLETE receipt stays in
        # the per-task `verification_receipts.jsonl` beside this row.
        **disclosed_list_projection(
            receipt.get("artifact_lifecycle"),
            key="artifact_lifecycle",
            limit=50,
            item=_lifecycle_row,
        ),
        # Also a path SET, so it goes through the SAME canonical derivation as the
        # identity paths above — de-duplicated on the RAW text BEFORE rendering, so the
        # bound is spent on distinct paths and `_omitted` counts what was really left out.
        **disclosed_list_projection(
            canonical_path_set(receipt.get("artifacts_missing_after")),
            key="artifacts_missing_after", limit=50,
        ),
        # v6.52.2: exit-masking sensor flag — the check's shell pipeline can launder the
        # real exit code (e.g. `... | tail`, `|| true`). FLAG-ONLY (status unchanged).
        "check_exit_masking": bool(receipt.get("check_exit_masking")),
        **disclosed_list_projection(
            receipt.get("check_exit_masking_reasons"), key="check_exit_masking_reasons", limit=10,
        ),
        # v6.54.4 criterion provenance (flag-only): task_stated | agent_defined.
        "criterion_source": str(receipt.get("criterion_source") or ""),
        "criterion_basis": truncate_review_artifact(receipt.get("criterion_basis"), limit=500),
    }


def read_receipts(path: Path) -> List[Dict[str, Any]]:
    """Read valid object rows from an append-only verification receipt file."""
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _outstanding(
    receipts: List[Dict[str, Any]],
    *,
    is_candidate: Callable[[Dict[str, Any]], bool],
    is_reconciler: Callable[[Dict[str, Any]], bool],
    reconciles: Callable[[Dict[str, Any], Dict[str, Any]], bool],
    same_identity: Callable[[Dict[str, Any], Dict[str, Any]], bool],
) -> List[Dict[str, Any]]:
    """EVERY still-outstanding verification IDENTITY, oldest evidence first — each
    candidate scanned against ALL the receipts that follow IT, then collapsed onto the
    typed identity KEY it names.

    "Collapse onto the identity it names" is only well defined because ``same_identity``
    is an EQUIVALENCE (key equality — see ``ReceiptIdentity.key``). Under the fallback
    CHAIN it replaced, sameness was not transitive, so which receipts merged and which
    survived depended on the order they were collected in; the outcome is now a function
    of the keys alone.

    One entry per IDENTITY, not per ROW (round 4): two consecutive failures of the same
    criterion / check / path with no green between them are ONE red verification, and
    reporting ``unreconciled_red_count=2`` for it told the reviewer there were two
    distinct problems to chase. Each identity is represented by its MOST RECENT
    outstanding receipt — the freshest summary and returncode for that verification —
    and ordered by that receipt, so ``latest_*`` really is the newest outstanding
    evidence. Receipts carrying no identity at all are never collapsed into each other:
    they are indistinguishable, not known-equal, and merging them would UNDERCOUNT.

    "Is anything still outstanding?" is a question about a SET of identities, and a
    single latest-POINTER cannot represent it (v6.78.0 review round 2): tracking one
    latest candidate lets a NEWER candidate erase an older still-outstanding one, after
    which the newer one's own clean re-grounding reports nothing outstanding at all —
    fail check A, fail check B, pass check B answered "no red" while A was still red;
    masked c1, masked c2, cleanly-reconciled c2 lost masked c1 the same way. Both flags
    then told the acceptance reviewer and the finalization nudge that no verification
    was outstanding, which is the one answer that must be unrepresentable here.

    Computing the outstanding SET per candidate makes it so; the ``latest_*`` helpers
    are then just its newest element, and callers that need the whole picture (the
    acceptance ``verification_summary`` counts) read the list."""
    valid = [receipt for receipt in receipts if isinstance(receipt, dict)]
    outstanding: List[Dict[str, Any]] = []
    for index, receipt in enumerate(valid):
        if not is_candidate(receipt):
            continue
        if any(
            is_reconciler(later) and reconciles(later, receipt)
            for later in valid[index + 1:]
        ):
            continue
        for position, kept in enumerate(outstanding):
            if same_identity(receipt, kept):
                outstanding.pop(position)  # same verification, newer evidence
                break
        outstanding.append(receipt)
    return outstanding


def unreconciled_failed(
    receipts: List[Dict[str, Any]],
    reconciling_statuses: Collection[str] = RED_RECONCILING_STATUSES,
) -> List[Dict[str, Any]]:
    """Every still-red VERIFICATION (oldest evidence first, one entry per identity) with
    no LATER grounding carrying the SAME typed identity key (owner Q28=B). A green whose
    key differs — a different criterion, a different command, a different observed path
    set, or no key where the red had one — clears neither its own peers nor an unrelated
    red; two failures of the SAME key with no green between them are one red, not two.
    Advisory only — never a gate."""
    statuses = frozenset(reconciling_statuses)
    return _outstanding(
        receipts,
        is_candidate=lambda receipt: str(receipt.get("status") or "") == "fail",
        is_reconciler=lambda receipt: str(receipt.get("status") or "") in statuses,
        reconciles=_reconciles,
        same_identity=_same_verification,
    )


def unreconciled_masked(
    receipts: List[Dict[str, Any]],
    reconciling_statuses: Collection[str] = RED_RECONCILING_STATUSES,
) -> List[Dict[str, Any]]:
    """Every still-masked CRITERION (oldest evidence first, one entry per identity) with
    no LATER CLEAN grounding that reconciles it: the same ``criterion_id`` key when the
    masked receipt has one, else ANY later clean grounding (see ``_reconciles_masked`` —
    the red path's check-text identity cannot apply where the remediation necessarily
    changes the command text). Repeated masked runs of the SAME criterion are one open
    masked check. A receipt that is ITSELF masked never reconciles another."""
    statuses = frozenset(reconciling_statuses)
    return _outstanding(
        receipts,
        is_candidate=receipt_is_masked_pass,
        is_reconciler=lambda receipt: (
            str(receipt.get("status") or "") in statuses
            and not bool(receipt.get("check_exit_masking"))
        ),
        reconciles=_reconciles_masked,
        same_identity=_same_masked_verification,
    )


def latest_unreconciled_failed(
    receipts: List[Dict[str, Any]],
    reconciling_statuses: Collection[str] = RED_RECONCILING_STATUSES,
) -> Optional[Dict[str, Any]]:
    """The NEWEST still-unreconciled failed receipt, or ``None`` when the outstanding
    set is empty — a projection of ``unreconciled_failed``, never its own tracker."""
    outstanding = unreconciled_failed(receipts, reconciling_statuses)
    return outstanding[-1] if outstanding else None


def latest_unreconciled_masked(
    receipts: List[Dict[str, Any]],
    reconciling_statuses: Collection[str] = RED_RECONCILING_STATUSES,
) -> Optional[Dict[str, Any]]:
    """The NEWEST still-unreconciled masked pass, or ``None`` when the outstanding set
    is empty — a projection of ``unreconciled_masked``, never its own tracker."""
    outstanding = unreconciled_masked(receipts, reconciling_statuses)
    return outstanding[-1] if outstanding else None


def latest_agent_defined(receipts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the latest ungrounded agent-defined passing criterion, if any."""
    for receipt in reversed(receipts):
        if not isinstance(receipt, dict):
            continue
        if str(receipt.get("status") or "") not in ("pass", "observed"):
            continue
        if str(receipt.get("criterion_source") or "") != "agent_defined":
            return None
        if str(receipt.get("criterion_basis") or "").strip():
            return None
        return receipt
    return None


def latest_unreconciled_failed_receipt(receipts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Pure core: the most recent RED receipt (``status=="fail"``) with NO later genuine
    grounding receipt for the SAME verification (a passing run-kind check or an observed
    artifact — see ``RED_RECONCILING_STATUSES`` above; a later ``declared`` does NOT
    reconcile). Returns the failing receipt, or ``None``. Structural: the typed receipt
    status decides pass/fail, and identity is ONE typed key: the ``criterion_id`` when
    present, else the canonical ``check`` text, else the observed ``paths`` set (owner
    Q28=B, content-ADDRESSING — never a semantic keyword gate). Kind AND value must match,
    so a green of another check — or one that omits the id — no longer clears a red; a red
    with NO key at all keeps the older any-later-green rule. Advisory, never a gate.
    The NEWEST element of the OUTSTANDING SET (``unreconciled_failed``)
    — never a single latest-pointer, which a newer red would let erase an older still-red
    one. Shared SSOT by the finalize nudge and the acceptance verification_summary so the
    reconciliation rule lives in one place."""
    return latest_unreconciled_failed(receipts, RED_RECONCILING_STATUSES)


def latest_unreconciled_failed_verification(
    drive_root: Any, task_id: str,
    *, receipts: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Disk-backed wrapper of ``latest_unreconciled_failed_receipt`` — reads the task's
    durable receipts. Feeds the one-shot red-verification finalization nudge: finalizing over
    your own host-attested red is a self-contradiction (Bible P3/P12), distinct from the
    receipt_absent case."""
    from ouroboros.outcome_receipt_store import read_verification_receipts

    rows = receipts if isinstance(receipts, list) else read_verification_receipts(drive_root, task_id)
    return latest_unreconciled_failed_receipt(rows)


def latest_unreconciled_masked_pass(receipts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Pure core (v6.52.2): the most recent PASS receipt whose check can MASK the real exit code
    (``check_exit_masking`` flag from the verify sensor — e.g. ``... | tail``, ``|| true``), with
    NO later CLEAN (non-masked) grounding receipt (a pass/observed whose check is not masked).
    Returns the masked passing receipt, or ``None``. Identity is the ``criterion_id`` key when
    the masked receipt carries one, else ANY clean grounding reconciles: its own text
    identity is the MASKED command, which the remediation necessarily changes, so the red
    path's check-text rule would be unclearable (``_reconciles_masked``).
    The NEWEST element of the OUTSTANDING SET (``unreconciled_masked``),
    so a cleanly reconciled newer masked check no longer takes an older one with it.
    FLAG-driven (typed receipt field); advisory only. Shared SSOT by the finalize nudge and
    the acceptance verification_summary."""
    return latest_unreconciled_masked(receipts, RED_RECONCILING_STATUSES)


def latest_unreconciled_masked_verification(
    drive_root: Any, task_id: str,
    *, receipts: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Disk-backed wrapper of ``latest_unreconciled_masked_pass`` — feeds the one-shot ADVISORY
    masked-check finalization nudge (the agent may still finalize). Distinct from the red nudge:
    that fires on a RED check; this fires on a green check whose exit code may be laundered."""
    from ouroboros.outcome_receipt_store import read_verification_receipts

    rows = receipts if isinstance(receipts, list) else read_verification_receipts(drive_root, task_id)
    return latest_unreconciled_masked_pass(rows)


def latest_agent_defined_verification(
    drive_root: Any, task_id: str,
    *, receipts: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Newest verify receipt whose criterion was AGENT-DEFINED without a stated basis
    (v6.54.4) — feeds the one-shot advisory criterion-provenance nudge: the check
    passed, but the success criterion was synthesized by the agent, so the agent is
    asked once to confirm it is equivalent to what the task actually requires."""
    from ouroboros.outcome_receipt_store import read_verification_receipts

    rows = receipts if isinstance(receipts, list) else read_verification_receipts(drive_root, task_id)
    return latest_agent_defined(rows)
