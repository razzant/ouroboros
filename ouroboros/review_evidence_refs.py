"""D-Q5 reviewer evidence-ref resolution (leaf).

Extracted from ``review_evidence.py`` (v6.91 synthesis) so that module stays
under the hard module gate after the acceptance-stream growth.  This module is a
LEAF over plain dicts: the enumerable exhibit-key vocabulary of ONE built packet
and the exact-membership resolver.  It must never import
``ouroboros.review_evidence`` — that module re-exports every name here, so the
historical import site (``from ouroboros.review_evidence import
acceptance_evidence_ref_vocabulary, resolve_criteria_evidence_refs, ...``)
keeps resolving unchanged.  ``annotate_criteria_evidence_resolution`` stays in
``review_evidence`` (it consumes these via the re-exported module globals, which
keeps its fail-closed seam patchable per the D-Q5 tests).
"""

from __future__ import annotations

from typing import Any, Dict

# D-Q5 non-resolving basis kinds: the ref NAMES a real packet entry, but that
# entry is not host-attested support. Kept as a CLOSED set beside the vocabulary
# so the disclosure row can say WHICH entry was named (never a bare "") while the
# clean gate still refuses to count it as resolved evidence.
CLAIM_ID_UNSUPPORTED = "claim_id_unsupported"
# A receipt row the packet DOES carry whose own status is not a passing one
# (fail/declared) or whose expectation did not match. The claim-id rule already
# refuses to count such a receipt through its claim (`support_status !=
# "supported"`); the `verification_receipts[i]` index form must not be a bypass
# of the same rule, so the row is DISCLOSED by name and never resolves.
RECEIPT_NOT_PASSING = "verification_receipt_not_passing"
# A section the packet itself tags `agent_supplied` (agent_supplied,
# reasoning_notes, candidate_answers): the agent's OWN prose. Counting it would
# let a reviewer certify a clean PASS out of the task's own words — the same hole
# `claim_id_unsupported` closes, one level up.
AGENT_SUPPLIED_SECTION = "agent_supplied_section"
# The declared-intent container. The host attests these ARE the recorded contract
# — never that the work satisfies it — so naming the section that HOLDS the claims
# must not become the way around an unsupported claim id.
DECLARED_INTENT_SECTION = "declared_intent_section"
# A top-level section with no provenance tag at all (counters, manifests, a packet
# built outside the acceptance builder). Fail CLOSED: unknown attestation is not
# attestation.
UNATTESTED_SECTION = "unattested_section"
# A host-recorded section whose decision-bearing bytes were capped for the
# reviewer. It remains enumerable and dispatchable, but cannot resolve a clean
# acceptance criterion until the packet carries the complete section.
PARTIAL_SECTION = "partial"
NON_RESOLVING_BASIS_KINDS = frozenset({
    CLAIM_ID_UNSUPPORTED,
    RECEIPT_NOT_PASSING,
    AGENT_SUPPLIED_SECTION,
    DECLARED_INTENT_SECTION,
    UNATTESTED_SECTION,
    PARTIAL_SECTION,
})

# Section provenance tags (``__provenance__`` in the built packet) that make a
# top-level section a HOST-ATTESTED exhibit: the host's own record of what
# happened — its collected diff/receipt summary/attribution (`host_attested`),
# its recording of tool results (`tool_result`), its artifact manifest
# (`artifact`). Everything else is classified above.
HOST_ATTESTED_SECTION_PROVENANCE = frozenset({"host_attested", "tool_result", "artifact"})
DECLARED_INTENT_SECTIONS = frozenset({
    "task_contract", "acceptance_claims_source", "plan_claims_exhibit",
})

# D-Q5 fail-closed row: the host could not resolve this panel's refs at all. It
# carries the SAME `supported_evidence_resolves=False` the clean gate already
# reads for an unresolved ref, so a broken resolver degrades the clean bit
# instead of authorizing it. `criterion="*"` marks it as panel-wide, not a
# reviewer-named criterion.
_RESOLUTION_UNAVAILABLE_ROW = {
    "criterion": "*",
    "refs": [],
    "supported_evidence_resolves": False,
    "resolution_status": "host_resolution_unavailable",
}


def acceptance_evidence_ref_vocabulary(evidence: Any) -> Dict[str, str]:
    """The enumerable canonical exhibit keys of ONE already-built packet (D-Q5).

    Maps each valid reviewer ``evidence_ref`` string to its CLOSED basis kind
    (claim_id | claim_id_unsupported | obligation_id | artifact |
    verification_receipt | verification_receipt_not_passing | packet_section | partial |
    agent_supplied_section | declared_intent_section | unattested_section — a
    closed table per ref kind, like ``IDENTITY_KINDS``). Pure derivation over the
    packet dict: no filesystem reads, no re-execution (a machine comparison must
    never become a read oracle — v6.61.1). ``verification_receipts[i]`` ids are
    POSITIONAL within this packet build and are never compared across panels.
    Specific ids are registered before generic section names so a claim id can
    never be shadowed by a same-named section.

    A receipt ref enumerates the packet's OWN ``verification_receipts`` exhibit
    rows — never a bare ``verification_summary.count``, which minted a valid ref
    for a receipt the packet did not carry and the reviewer could not read. A row
    resolves only while it is GREEN (status pass/observed and not ``matched is
    False`` — the same predicate the host support table applies to claims); a
    red/declared receipt registers as ``verification_receipt_not_passing``,
    disclosed by name, because the claim-id rule below already refuses that
    receipt through its claim and the index form must not be a bypass.

    A claim id is EVIDENCE only through the host's own support table: the plan's
    binding is "claim id → host receipt identity when claims exist". So a claim
    resolves iff ``acceptance_support_refs`` (host_attested, built from the SAME
    claim set in this build) reports ``support_status == "supported"`` for it —
    i.e. a linked receipt actually passed. A claim that is only DECLARED, whose
    receipt failed, that has no receipt, or whose support table is absent
    altogether registers as ``claim_id_unsupported`` and cannot support a
    criterion: otherwise a reviewer citing a bare claim id would certify a clean
    PASS out of the task's own restated intent, which is precisely the fabricated
    -evidence hole D-Q5 exists to close.

    A top-level SECTION resolves on the same rule, read off the packet's own
    ``__provenance__`` table: only a HOST-ATTESTED exhibit counts
    (`host_attested` / `tool_result` / `artifact`). The sections the packet tags
    `agent_supplied` (reasoning_notes, candidate_answers, agent_supplied), the
    declared-intent container (`task_contract`, which HOLDS the claims), and any
    section with no provenance tag are DISCLOSED by name and never resolve —
    otherwise citing "reasoning_notes" or "task_contract" would buy exactly the
    clean PASS a bare unsupported claim id cannot."""
    ev = evidence if isinstance(evidence, dict) else {}
    vocab: Dict[str, str] = {}
    contract = ev.get("task_contract") if isinstance(ev.get("task_contract"), dict) else {}
    supported_claim_ids = {
        str(row.get("criterion_id") or "").strip()
        for row in (ev.get("acceptance_support_refs") or [])
        if isinstance(row, dict) and str(row.get("support_status") or "") == "supported"
    }
    for claim in contract.get("acceptance_claims") or []:
        if isinstance(claim, dict) and str(claim.get("id") or "").strip():
            claim_id = str(claim["id"]).strip()
            vocab.setdefault(
                claim_id,
                "claim_id" if claim_id in supported_claim_ids else CLAIM_ID_UNSUPPORTED,
            )
    for obligation in ev.get("acceptance_obligations") or []:
        if isinstance(obligation, dict) and str(obligation.get("id") or "").strip():
            vocab.setdefault(str(obligation["id"]).strip(), "obligation_id")
    for artifact in ev.get("artifacts") or []:
        if isinstance(artifact, dict) and str(artifact.get("name") or "").strip():
            vocab.setdefault(str(artifact["name"]).strip(), "artifact")
    receipt_rows = ev.get("verification_receipts") if isinstance(ev.get("verification_receipts"), list) else []
    for row in receipt_rows:
        if not isinstance(row, dict) or not str(row.get("ref") or "").strip():
            continue
        passing = (
            str(row.get("status") or "") in ("pass", "observed")
            and row.get("matched") is not False
        )
        vocab.setdefault(
            str(row["ref"]).strip(),
            "verification_receipt" if passing else RECEIPT_NOT_PASSING,
        )
    provenance = ev.get("__provenance__") if isinstance(ev.get("__provenance__"), dict) else {}
    for key in ev:
        name = str(key)
        if name.startswith("__"):
            continue
        tag = str(provenance.get(name) or "")
        if name == "tool_trajectory" and (
            ev.get("tool_trajectory_complete") is False
            or bool(ev.get("tool_trajectory_omitted_leading"))
            or any(
                isinstance(row, dict) and row.get("result_complete") is False
                for row in (ev.get(name) if isinstance(ev.get(name), list) else [])
            )
        ):
            basis = PARTIAL_SECTION
        elif name == "repo_diff" and ev.get("repo_diff_complete") is False:
            basis = PARTIAL_SECTION
        elif name == "skill_lifecycle" and ev.get("skill_lifecycle_complete") is False:
            basis = PARTIAL_SECTION
        elif name in DECLARED_INTENT_SECTIONS:
            basis = DECLARED_INTENT_SECTION
        elif tag in HOST_ATTESTED_SECTION_PROVENANCE:
            basis = "packet_section"
        elif tag == "agent_supplied":
            basis = AGENT_SUPPLIED_SECTION
        else:
            basis = UNATTESTED_SECTION
        vocab.setdefault(name, basis)
    return vocab


def resolve_criteria_evidence_refs(criteria: Any, vocabulary: Dict[str, str]) -> list:
    """EXACT-membership resolution of reviewer ``evidence_refs`` for SUPPORTED
    criteria (D-Q5). No substring/fuzzy matching — a non-injective comparison
    lets a fabricated ref 'resolve' against an unrelated exhibit (the v6.78
    lossy-identity lesson). Returns disclosure rows ONLY for supported criteria
    carrying at least one unresolved ref; each row names the deciding basis for
    every ref (rounds-6/7 rule: whatever decides is what is reported) plus
    ``supported_evidence_resolves`` — whether at least ONE ref resolved, which is
    all ``task_acceptance_is_clean`` consumes. A basis in
    ``NON_RESOLVING_BASIS_KINDS`` (for example, an unsupported claim id or a
    partial section) is DISCLOSED by name but does not resolve. Never touches
    parse validity, quorum, or verdicts (the v6.71.1 starvation class stays
    closed)."""
    rows: list = []
    for item in criteria if isinstance(criteria, list) else []:
        if not isinstance(item, dict):
            continue
        if str(item.get("status") or "").strip().lower() != "supported":
            continue
        raw = item.get("evidence_refs")
        refs = raw if isinstance(raw, list) else ([raw] if raw else [])
        projected = []
        resolved_any = False
        unresolved_count = 0
        for ref in refs:
            text = str(ref or "").strip()
            basis = vocabulary.get(text, "") if text else ""
            if basis and basis not in NON_RESOLVING_BASIS_KINDS:
                resolved_any = True
            else:
                unresolved_count += 1
            projected.append({"ref": text[:200], "resolved_as": basis})
        if unresolved_count:
            rows.append({
                "criterion": str(item.get("criterion") or "")[:300],
                "refs": projected,
                "supported_evidence_resolves": resolved_any,
            })
    return rows
