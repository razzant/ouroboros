"""Plan spec object, typed findings, host aggregate and the ONE structural path
fact for ``plan_task`` — pure functions only (no I/O, no ToolContext). The
evidence manifest lives in the companion ``ouroboros.tools.plan_evidence``,
the reviewer packet builders in ``ouroboros.tools.plan_packet``.

Design principle (BIBLE P5, owner-approved 2026-08-15): the HOST checks shape,
identity, membership, bounds and ONE structural path fact (``constitutional``);
it never interprets the domain. Everything here works identically for a code
change, a slide deck, a literature review, a GUI flow or a trip plan — a spec
with ZERO file paths is first-class.

Bounds (every cut is disclosed, never silent — DEVELOPMENT.md "No silent
truncation"; list bounds record an omission entry, string bounds go through the
SSOT ``utils.truncate_review_artifact`` marker):

* ``MAX_LIST_ITEMS`` — items kept per spec list (in_scope, non_goals,
  invariants, decisions, deferred, affected_resources, evidence,
  acceptance_claims); ``MAX_REJECTED_PER_DECISION`` — nested ``decision.rejected``.
* ``MAX_ITEM_CHARS`` / ``MAX_GOAL_CHARS`` — per-string bounds (same 600-char
  default as ``task_contract._bounded_claim_text``).
* ``MAX_FINDINGS_PER_SLOT`` / ``MAX_FINDING_TEXT_CHARS`` — reviewer findings
  kept per slot and per summary/recommendation.
* ``PACKET_*_CHARS`` — reviewer-packet section bounds (objective, plan prose,
  root exploration log; SPEC and prior cycles are bounded STRUCTURALLY by
  ``bounded_json`` — whole items with disclosed counts + full-set hash).
"""

from __future__ import annotations

from hashlib import sha256
import json
import pathlib
import re
from typing import Any, Callable, Iterable, Mapping, Optional

from ouroboros.config import adaptive_quorum
from ouroboros.contracts.task_contract import normalize_acceptance_claims
from ouroboros.tool_access import path_is_relative_to
from ouroboros.triad_review import empty_array_is_verified_clean, extract_json_array
from ouroboros.utils import truncate_review_artifact

MAX_LIST_ITEMS = 40
MAX_REJECTED_PER_DECISION = 8
MAX_ITEM_CHARS = 600
MAX_GOAL_CHARS = 2000
MAX_FINDINGS_PER_SLOT = 32
# Per-task `need_evidence` memory: reviewers' requests the host remembers (and, W3, attaches).
# Bounded so the durable review state stays bounded whatever the panel asks for; a request past
# the cap is demoted (never remembered), disclosed `need_evidence_memory_full`.
MAX_NEED_EVIDENCE_MEMORY = 4 * MAX_LIST_ITEMS
MAX_FINDING_TEXT_CHARS = 2000
PACKET_OBJECTIVE_CHARS = 8_000
PACKET_SPEC_CHARS = 120_000
PACKET_PRIOR_FINDING_SUMMARY_CHARS = 400
PACKET_PROSE_CHARS = 40_000
PACKET_EXPLORATION_CHARS = 12_000
PACKET_PRIOR_CYCLES_CHARS = 60_000

FINDING_CLASSES = ("blocking", "note", "need_evidence")
AGGREGATES = ("GREEN", "REVIEW_REQUIRED", "REVISE_PLAN", "DEGRADED")
DISPOSITION_DECISIONS = ("accept", "reject", "defer")

_SPEC_STRING_LISTS = ("in_scope", "non_goals", "invariants", "affected_resources", "evidence")
_SPEC_KEYS = frozenset({
    "goal", "in_scope", "non_goals", "acceptance_claims", "invariants",
    "decisions", "deferred", "affected_resources", "evidence",
})
_URL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
_FILE_SCHEME = "file://"
_TASK_LOCATOR_PREFIX = "task:"


# --------------------------------------------------------------------------- spec


def bounded_text(value: Any, limit: int) -> str:
    """Strip + SSOT disclosed truncation (marker visible, anti-waste floor kept)."""
    return truncate_review_artifact(("" if value is None else str(value)).strip(), limit=limit)


MAX_ID_CHARS = 80


def _sanitize_id(raw: Any, fallback: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in str(raw or "").strip())[:MAX_ID_CHARS]
    return text or fallback


def _unique_id(candidate: str, seen: set[str]) -> str:
    base, suffix, out = candidate, 2, candidate
    while out in seen:
        out = f"{base}_{suffix}"
        suffix += 1
    seen.add(out)
    return out


def _cap_list(items: list, label: str, omissions: list[str], *, bound: int = MAX_LIST_ITEMS) -> list:
    """Bound a list at ``bound`` and RECORD the cut (P1) — never a silent slice."""
    if len(items) <= bound:
        return items
    omissions.append(f"{label}: {len(items)} items declared, kept the first {bound} (bound {bound})")
    return items[:bound]


def _is_scalar(value: Any) -> bool:
    """str/int/float are tolerated as text; bool is not (``True`` is not a spec item)."""
    return isinstance(value, (str, int, float)) and not isinstance(value, bool)


def _string_list(raw: Any, label: str, errors: list[str], omissions: list[str]) -> list[str]:
    """Tolerant string list: bare string → one item; blanks dropped; bounded with disclosure."""
    if raw is None:
        return []
    items = [raw] if _is_scalar(raw) else raw
    if not isinstance(items, list):
        errors.append(f"{label}: must be a list of strings")
        return []
    out: list[str] = []
    blank = 0
    for index, item in enumerate(items):
        if item is None:
            continue
        if not _is_scalar(item):
            errors.append(f"{label}[{index}]: must be a string")
            continue
        text = bounded_text(item, MAX_ITEM_CHARS)
        if text:
            out.append(text)
        else:
            # I-20: every cut in this module is disclosed; a blank item is a cut too — as ONE
            # bounded disclosure per list, so the disclosure list itself cannot outgrow the state.
            blank += 1
    if blank:
        omissions.append(f"{label}: {blank} blank item(s) dropped")
    return _cap_list(out, label, omissions)


def _object_list(raw: Any, label: str, text_key: str, errors: list[str]) -> list[dict]:
    """Tolerant object list: bare string → {text_key: string}; non-object → error."""
    if raw is None:
        return []
    items = [raw] if isinstance(raw, (str, Mapping)) else raw
    if not isinstance(items, list):
        errors.append(f"{label}: must be a list of objects")
        return []
    out: list[dict] = []
    for index, item in enumerate(items):
        if isinstance(item, Mapping):
            out.append(dict(item))
        elif isinstance(item, str):
            out.append({text_key: item})
        else:
            errors.append(f"{label}[{index}]: must be an object or a string")
    return out


def _normalize_claims(raw: Any, errors: list[str], omissions: list[str], seen: set[str]) -> list[dict]:
    if raw is None:
        raw = []
    elif isinstance(raw, (str, Mapping)):
        raw = [raw]
    if not isinstance(raw, list):
        errors.append("acceptance_claims: must be a list")
        return []
    claims = normalize_acceptance_claims(raw)  # SSOT schema/normalizer (task_contract)
    dropped = len(raw) - len(claims)
    if dropped > 0:
        omissions.append(
            f"acceptance_claims: {dropped} item(s) dropped as empty/invalid by the "
            "task_contract acceptance-claims normalizer"
        )
    claims = _cap_list(claims, "acceptance_claims", omissions)
    for index, claim in enumerate(claims, start=1):
        # Ids are HOST-MINTED positionally (claim_1..N) because they are the only valid
        # `breaks` targets a reviewer may name: a caller-chosen id could shadow another
        # element's id or move under the reviewer's feet between cycles. A declared id is
        # kept beside it so nothing is lost and receipts can still be matched by text.
        declared = str(claim.get("id") or "").strip()
        minted = _unique_id(f"claim_{index}", seen)
        if declared and declared != minted:
            claim["declared_id"] = declared
        claim["id"] = minted
    return claims


def _normalize_decisions(raw: Any, errors: list[str], omissions: list[str], seen: set[str]) -> list[dict]:
    out: list[dict] = []
    items = _cap_list(_object_list(raw, "decisions", "choice", errors), "decisions", omissions)
    for index, item in enumerate(items, start=1):
        choice = bounded_text(item.get("choice"), MAX_ITEM_CHARS)
        if not choice:
            errors.append(f"decisions[{index - 1}]: choice is required")
            continue
        out.append({
            # host-minted, positional: a caller-chosen id could shadow another element's
            # id or move between cycles, and these ids are what a blocking finding names.
            "id": _unique_id(f"decision_{index}", seen),
            "choice": choice,
            "rejected": _cap_list(
                _string_list(item.get("rejected"), f"decisions[{index - 1}].rejected", errors, omissions),
                f"decisions[{index - 1}].rejected", omissions, bound=MAX_REJECTED_PER_DECISION,
            ),
            "why": bounded_text(item.get("why"), MAX_ITEM_CHARS),
        })
    return out


def _normalize_deferred(raw: Any, errors: list[str], omissions: list[str], seen: set[str]) -> list[dict]:
    out: list[dict] = []
    items = _cap_list(_object_list(raw, "deferred", "what", errors), "deferred", omissions)
    for index, item in enumerate(items, start=1):
        what = bounded_text(item.get("what"), MAX_ITEM_CHARS)
        if not what:
            errors.append(f"deferred[{index - 1}]: what is required")
            continue
        out.append({
            # host-minted, positional: a caller-chosen id could shadow another element's
            # id or move between cycles, and these ids are what a blocking finding names.
            "id": _unique_id(f"deferred_{index}", seen),
            "what": what,
            "why_safe_to_defer": bounded_text(item.get("why_safe_to_defer"), MAX_ITEM_CHARS),
        })
    return out


def normalize_spec(raw: Mapping[str, Any] | None) -> tuple[dict, list[str]]:
    """Normalize a plan spec (plan §9.2 schema) → ``(spec, errors)``.

    Tolerant of strings-vs-dicts, mints stable ids (claim_N / invariant_N /
    decision_N / deferred_N the declared id, when different, is kept beside it as `declared_id`), trims, and bounds
    every list at ``MAX_LIST_ITEMS`` — the excess is RECORDED under
    ``spec["normalization_omissions"]`` (P1), never silently dropped. Genuinely
    malformed input (missing goal, unknown field, wrong container type, decision
    without choice) yields typed error strings; on any error the returned spec
    is best-effort and NOT authoritative — the caller must refuse it.
    Invariant ids are positional (``invariant_1..N``) because the schema keeps
    ``invariants: [str]``; ``spec_ids``/``spec_with_ids`` derive them; the id
    ``goal`` is reserved for the intention as a whole (D32).
    """
    errors: list[str] = []
    omissions: list[str] = []
    if not isinstance(raw, Mapping):
        return {}, ["spec: must be an object"]
    unknown = sorted(str(key) for key in raw if key not in _SPEC_KEYS)
    if unknown:
        errors.append("spec: unknown fields: " + ", ".join(unknown))
    goal = bounded_text(raw.get("goal"), MAX_GOAL_CHARS) if isinstance(raw.get("goal"), str) else ""
    if not goal:
        errors.append(
            "goal: must be a string" if raw.get("goal") is not None and not isinstance(raw.get("goal"), str)
            else "goal: required non-empty string"
        )
    spec: dict[str, Any] = {"goal": goal}
    for key in _SPEC_STRING_LISTS:
        spec[key] = _string_list(raw.get(key), key, errors, omissions)
    seen = {GOAL_ID, *(f"invariant_{i}" for i in range(1, len(spec["invariants"]) + 1))}
    spec["acceptance_claims"] = _normalize_claims(raw.get("acceptance_claims"), errors, omissions, seen)
    spec["decisions"] = _normalize_decisions(raw.get("decisions"), errors, omissions, seen)
    spec["deferred"] = _normalize_deferred(raw.get("deferred"), errors, omissions, seen)
    spec["normalization_omissions"] = omissions
    ordered = {key: spec[key] for key in (
        "goal", "in_scope", "non_goals", "acceptance_claims", "invariants", "decisions",
        "deferred", "affected_resources", "evidence", "normalization_omissions",
    )}
    return ordered, errors


GOAL_ID = "goal"  # D32: the intention as a whole is always a valid `breaks` target


def spec_ids(spec: Mapping[str, Any]) -> frozenset[str]:
    """Every id a finding's ``breaks`` may name: ``goal`` (always — owner decision D32,
    so a claims-less spec can still receive a blocking finding) plus
    claim_/invariant_/decision_/deferred_ ids."""
    ids: set[str] = {GOAL_ID}
    for key in ("acceptance_claims", "decisions", "deferred"):
        for item in spec.get(key) or []:
            if isinstance(item, Mapping) and str(item.get("id") or ""):
                ids.add(str(item["id"]))
    ids.update(f"invariant_{i}" for i in range(1, len(spec.get("invariants") or []) + 1))
    return frozenset(ids)


def spec_with_ids(spec: Mapping[str, Any]) -> dict:
    """The spec as the reviewer sees it: positional invariants carry their ids inline."""
    view = {key: value for key, value in spec.items()}
    view["goal"] = {"id": GOAL_ID, "text": spec.get("goal") or ""}
    view["invariants"] = [
        {"id": f"invariant_{i}", "text": text}
        for i, text in enumerate(spec.get("invariants") or [], start=1)
    ]
    return view


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(encoded.encode("utf-8")).hexdigest()


def spec_hash(spec: Mapping[str, Any]) -> str:
    """sha256 over canonical JSON (sorted keys, compact separators): stable under key order/whitespace."""
    return _canonical_hash(dict(spec))


def _by_id(items: Any) -> dict[str, dict]:
    return {
        str(item["id"]): dict(item)
        for item in (items or []) if isinstance(item, Mapping) and item.get("id")
    }


def _text_by_id(spec: Mapping[str, Any]) -> dict[str, str]:
    """id → identifying text for every id-bearing element (invariants positional)."""
    out = {f"invariant_{i}": t for i, t in enumerate(spec.get("invariants") or [], start=1)}
    for key, text_key in (("acceptance_claims", "claim"), ("decisions", "choice"), ("deferred", "what")):
        for item in spec.get(key) or []:
            if isinstance(item, Mapping) and item.get("id"):
                out[str(item["id"])] = str(item.get(text_key) or "")
    return out


def _renumbered(prev: Mapping[str, Any], spec: Mapping[str, Any]) -> list[dict]:
    """Same identifying text under a DIFFERENT id between cycles → ``[{from, to, text}]``.
    Ids are positional/minted, so an agent dropping or reordering one element shifts the
    later ones; the reviewer must re-target ``breaks`` against the CURRENT ids (B-02)."""
    before, after = _text_by_id(prev), _text_by_id(spec)
    after_by_text: dict[str, list[str]] = {}
    for fid, text in after.items():
        after_by_text.setdefault(text, []).append(fid)
    out: list[dict] = []
    for old_id, text in before.items():
        if not text or after.get(old_id) == text:
            continue
        for new_id in after_by_text.get(text, []):
            if new_id != old_id and before.get(new_id) != text:
                out.append({"from": old_id, "to": new_id, "text": text})
                break
    return out


def spec_delta(prev_spec: Mapping[str, Any] | None, spec: Mapping[str, Any]) -> dict:
    """What changed between two normalized specs (wave ≥2 packets).

    Id-bearing collections (acceptance_claims, decisions, deferred) diff by id
    (added / removed / changed content); plain string lists diff by text; the
    goal is a changed flag; ``renumbered`` lists elements whose text moved to a
    different id (positional ids shift when an earlier element is dropped or
    reordered). ``prev_spec=None`` means everything is added.
    """
    prev = dict(prev_spec or {})
    ids: dict[str, list[str]] = {"added": [], "removed": [], "changed": []}
    for key in ("acceptance_claims", "decisions", "deferred"):
        before, after = _by_id(prev.get(key)), _by_id(spec.get(key))
        ids["added"].extend(sorted(set(after) - set(before)))
        ids["removed"].extend(sorted(set(before) - set(after)))
        ids["changed"].extend(sorted(k for k in set(before) & set(after) if before[k] != after[k]))
    lists: dict[str, dict[str, list[str]]] = {}
    for key in _SPEC_STRING_LISTS:
        before_l, after_l = list(prev.get(key) or []), list(spec.get(key) or [])
        lists[key] = {
            "added": [t for t in after_l if t not in before_l],
            "removed": [t for t in before_l if t not in after_l],
        }
    goal_changed = str(prev.get("goal") or "") != str(spec.get("goal") or "")
    changed = goal_changed or any(ids.values()) or any(
        entry["added"] or entry["removed"] for entry in lists.values()
    )
    return {
        "changed": changed,
        "goal_changed": goal_changed,
        "ids": ids,
        "lists": lists,
        "renumbered": _renumbered(prev, spec),
        "prev_hash": spec_hash(prev) if prev_spec is not None else "",
        "hash": spec_hash(spec),
    }


def _lists_by_size(node: Any, path: str = "") -> list[tuple[str, list]]:
    """Every list reachable through dict nesting (not through list items), largest first."""
    found: list[tuple[str, list]] = []
    if isinstance(node, list):
        found.append((path or "$", node))
    elif isinstance(node, dict):
        for key, value in node.items():
            found.extend(_lists_by_size(value, f"{path}.{key}" if path else str(key)))
    return sorted(found, key=lambda entry: -len(_dumps(entry[1])))


def _dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def _set_at(node: Any, path: str, value: list) -> None:
    keys = path.split(".")
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value


def bounded_json(payload: Any, limit: int) -> tuple[str, list[str]]:
    """Serialize ``payload`` within ``limit`` chars WITHOUT clipping JSON mid-string
    (DEVELOPMENT "No silent truncation" for lists): oversize is reduced at the
    STRUCTURE level — whole items are dropped from the largest list(s), each cut
    disclosed as ``<path>: kept K of N items; full-set sha256=<h>``. If the payload
    is still oversized with every list emptied, a typed omission object replaces it
    (never clipped JSON). Returns ``(json_text, omission_notes)``. Deep-copied via JSON."""
    work = json.loads(json.dumps(payload, default=str))
    notes: list[str] = []
    text = _dumps(work)
    while len(text) > limit:
        candidates = [(path, items) for path, items in _lists_by_size(work) if items]
        if not candidates:
            full_hash = _canonical_hash(payload)
            return _dumps({
                "omitted": True, "reason": "oversized_after_structural_bounding",
                "chars": len(text), "full_payload_sha256": full_hash,
            }), notes + [f"payload omitted: {len(text)} chars exceed the {limit}-char bound; sha256={full_hash}"]
        path, items = candidates[0]
        need = len(text) - limit
        kept = list(items)
        dropped = 0
        while kept and dropped < need:
            dropped += len(_dumps(kept.pop())) + 2
        if path == "$":
            work = kept
        else:
            _set_at(work, path, kept)
        notes.append(
            f"{path}: kept {len(kept)} of {len(items)} items; full-set sha256={_canonical_hash(items)}"
        )
        text = _dumps(work)
    return text, notes


# ------------------------------------------------------------------ constitutional


def _is_url(locator: str) -> bool:
    return bool(_URL_RE.match(locator)) and not locator.startswith(_FILE_SCHEME)


def _is_path_locator(locator: str) -> bool:
    return bool(locator) and not _is_url(locator) and not locator.startswith(_TASK_LOCATOR_PREFIX)


def _resolve_locator_path(locator: str, root: pathlib.Path) -> tuple[Optional[pathlib.Path], str]:
    """Relative → under ``root``; absolute (or ``file://`` absolute) as-is. Returns
    ``(path, "")`` or ``(None, reason)`` — ``symlink_loop`` (RuntimeError from resolve) or
    ``unsupported_locator`` (NUL bytes, over-long components, …). Never raises."""
    if locator.startswith(_FILE_SCHEME):
        locator = locator[len(_FILE_SCHEME):]
    try:
        candidate = pathlib.Path(locator)
        return (candidate if candidate.is_absolute() else root / candidate).resolve(strict=False), ""
    except RuntimeError:
        return None, "symlink_loop"
    except (OSError, ValueError):
        return None, "unsupported_locator"


def _under(path: pathlib.Path, root: pathlib.Path) -> bool:
    """``tool_access.path_is_relative_to`` guarded against symlink loops on re-resolution."""
    try:
        return path_is_relative_to(path, root)
    except RuntimeError:
        return False


def _default_exists(path: pathlib.Path) -> bool:
    """Fail CLOSED: only a definite "this path is not there" lets an evidence locator skip
    the constitutional escalation. An unreadable path (permissions, a transient I/O error)
    counts as present, because the safe error is carrying the constitution needlessly, never
    dropping it on a filesystem hiccup."""
    try:
        return path.exists()
    except (OSError, ValueError, RuntimeError):
        return True


def resolve_constitutional(
    *,
    active_root: str | pathlib.Path,
    system_repo_root: str | pathlib.Path,
    affected_resources: Iterable[str],
    evidence: Iterable[str],
    payload_roots: Iterable[str | pathlib.Path] = (),
    evidence_exists: Optional[Callable[[pathlib.Path], bool]] = None,
) -> tuple[bool, str]:
    """ONE structural fact: does this plan touch Ouroboros's own body?

    Port of ``plan_review_runtime.resolve_plan_class`` without the enum. True iff
    any ``affected_resources`` / ``evidence`` PATH locator (relative resolved
    against ``active_root``, absolute or ``file://`` as-is) resolves to or under
    the system repository. The active binding alone does NOT decide (owner
    decision D29: a plan bound to the system repo that declares no system path
    is not constitutional). Skill-payload paths under ``payload_roots`` are
    exempt (data plane, as today). URLs and ``task:`` locators never make it
    true. Returns ``(constitutional, note)`` — the note names the deciding
    locator for disclosure.
    """
    system = pathlib.Path(system_repo_root).resolve(strict=False)
    active = pathlib.Path(active_root).resolve(strict=False)
    # Payload roots are supplied by the FROZEN skill-payload predicate
    # (``plan_review_runtime.plan_payload_roots`` -> ``resolve_skill_payload_target``),
    # never by the plan text, so a caller cannot dress an arbitrary system path as a
    # payload. They intentionally survive even when they resolve INSIDE the system
    # repo: a skill task whose active workspace IS the repo still touches only
    # data-plane payload files (frozen behaviour, task 7881ad77902d4d25). A reviewer
    # proposal to drop system-repo-nested payload roots (S-B05) was REJECTED for that
    # reason; the residual it names requires control of the payload predicate itself.
    payloads = [pathlib.Path(p).resolve(strict=False) for p in payload_roots]
    exists = evidence_exists if evidence_exists is not None else _default_exists
    skipped: list[str] = []
    for label, locators in (("affected_resources", affected_resources), ("evidence", evidence)):
        for raw in locators or []:
            locator = str(raw or "").strip()
            if not _is_path_locator(locator):
                continue
            resolved, _reason = _resolve_locator_path(locator, active)
            if resolved is None or any(_under(resolved, payload) for payload in payloads):
                continue
            if resolved == system or _under(resolved, system):
                # An `affected_resources` target counts whether or not it exists yet — creating
                # `ouroboros/new_module.py` IS self-modification. An `evidence` locator is only
                # something to LOOK AT: a path that does not exist (a typo, a file belonging to
                # another workspace) must not drag the constitutional pack in behind it — but the
                # skip is DISCLOSED, never reported as "no locator resolved" (P1).
                if label == "evidence" and not exists(resolved):
                    skipped.append(locator)
                    continue
                return True, (
                    f"constitutional: {label} locator {locator!r} resolves under the "
                    "Ouroboros system repository (structural fact)"
                )
    if skipped:
        return False, (
            "not constitutional: the only system-repo locators declared are EVIDENCE paths that do "
            f"not exist ({', '.join(repr(item) for item in skipped[:5])}) — declare them under "
            "affected_resources if the work will change them"
        )
    return False, "not constitutional: no declared locator resolves under the Ouroboros system repository"


# ----------------------------------------------------------------------- findings

PLAN_REVIEW_SESSION_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["findings"],
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "class", "summary"],
                "properties": {
                    "id": {"type": "string"},
                    "class": {"type": "string", "enum": ["blocking", "note", "need_evidence"]},
                    "breaks": {"type": "string"},
                    "locator": {"type": "string"},
                    "summary": {"type": "string"},
                    "recommendation": {"type": "string"},
                },
            },
        },
    },
}


_PLAN_FINDING_ELEMENT_SCHEMA = """\
{
  "id": "<short local id, e.g. f1>",
  "class": "blocking" | "note" | "need_evidence",
  "breaks": "<spec id — REQUIRED for blocking: goal | claim_N | invariant_N | decision_N | deferred_N>",
  "locator": "<REQUIRED for need_evidence: the exact path / URL / task:<id> you need>",
  "summary": "<what is wrong or missing, concretely>",
  "recommendation": "<the smallest change to the SPEC that resolves it>"
}"""

# Same protocol family as triad_review.REVIEW_JSON_ARRAY_CONTRACT: bare array,
# optional single code fence, NO_FINDINGS sentinel for a verified-clean review.
PLAN_FINDINGS_ARRAY_CONTRACT = f"""\
Return ONLY a JSON array of findings, optionally inside one code fence. Each element:
{_PLAN_FINDING_ELEMENT_SCHEMA}
If you reviewed everything and found NOTHING to report, return the empty array
followed by the sentinel word NO_FINDINGS on its own line, and nothing else.
An EMPTY array surrounded by prose, or the sentinel without the array, is a
non-response and excluded from quorum. There is no findings quota.
"""


def parse_findings(text: str) -> tuple[list[dict], Optional[str]]:
    """Reviewer output → ``(raw_findings, parse_error)`` via the repository's
    findings-array protocol (``triad_review.extract_json_array`` /
    ``empty_array_is_verified_clean``). ``[]`` + ``NO_FINDINGS`` → clean;
    prose-only, unparseable, non-object elements, or a bare ``[]`` inside prose
    (indistinguishable from a refusal) → ``parse_error`` (excluded from quorum).
    A NON-empty array embedded in prose is accepted (inherited from
    ``extract_json_array``'s best-effort scan, same as the triad).
    """
    raw = str(text or "")
    if empty_array_is_verified_clean(raw):
        return [], None
    parsed = extract_json_array(
        raw, validate_fn=lambda items: all(isinstance(item, Mapping) for item in items),
    )
    if parsed is None:
        return [], "no findings JSON array found (prose-only or unparseable response)"
    if not parsed:
        return [], "empty findings array without the NO_FINDINGS sentinel (or with surrounding prose)"
    return [dict(item) for item in parsed], None


def validate_findings(
    findings: Iterable[Mapping[str, Any]],
    *,
    spec_ids: Iterable[str],
    seen_locators: Iterable[str],
    slot: str = "",
) -> tuple[list[dict], list[str], frozenset[str]]:
    """Structural validation of one slot's findings → ``(normalized, disclosures, seen_after)``.

    Host checks membership/shape only (P5): ``blocking`` needs ``breaks`` ∈
    ``spec_ids`` else DEMOTED to ``note`` (``blocking_without_valid_breaks``);
    ``need_evidence`` needs a non-empty ``locator`` else demoted; a missing
    ``summary`` is filled with ``(missing summary)`` + disclosure (a finding is
    never dropped — an ok slot must not launder its blocking finding away); a
    ``need_evidence`` locator already in ``seen_locators`` — the PER-TASK
    (cross-cycle) memory the caller persists in ``plan_review_state`` — or
    repeated within this slot is DEMOTED to ``note`` (``need_evidence_repeat``):
    the host never re-attaches it, but the finding stays in the aggregate so a
    re-asked question cannot close the wave by disappearing; ids are
    minted ``f{slot}_{n}`` when missing; the slot is capped at
    ``MAX_FINDINGS_PER_SLOT`` with an omission disclosure. ``seen_after`` is the
    updated locator memory for the caller to persist; the input is not mutated.
    The engine validates the slots of ONE wave sequentially against the CUMULATIVE
    memory (it passes the running ``seen_after`` back in), so the per-task memory
    cap ``MAX_NEED_EVIDENCE_MEMORY`` is exact across slots; a second slot asking
    for a locator the first already requested is a `need_evidence_repeat` note —
    one request suffices, the wave stays open the same way.
    """
    ids = frozenset(str(s) for s in spec_ids)
    seen = set(str(s) for s in seen_locators)
    disclosures: list[str] = []
    normalized: list[dict] = []
    local_ids: set[str] = set()
    items = [item for item in findings]
    if len(items) > MAX_FINDINGS_PER_SLOT:
        disclosures.append(
            f"findings_capped:{MAX_FINDINGS_PER_SLOT}/{len(items)} (bound MAX_FINDINGS_PER_SLOT)"
        )
        items = items[:MAX_FINDINGS_PER_SLOT]
    for index, item in enumerate(items, start=1):
        if not isinstance(item, Mapping):
            disclosures.append(f"finding_{index}_not_object")
            continue
        fid = _unique_id(_sanitize_id(item.get("id"), f"f{slot}_{index}"), local_ids)
        summary = bounded_text(item.get("summary"), MAX_FINDING_TEXT_CHARS)
        if not summary:
            # Never DROP a finding: a blocking finding without a summary still blocks (S-B03).
            disclosures.append(f"missing_summary:{fid}")
            summary = "(missing summary)"
        klass = str(item.get("class") or "").strip().lower()
        breaks = str(item.get("breaks") or "").strip()[:MAX_ID_CHARS]  # an id, bounded like ids
        locator = str(item.get("locator") or "").strip()
        if klass not in FINDING_CLASSES:
            disclosures.append(f"unknown_class:{fid}:{klass or '<empty>'}")
            klass = "note"
        if klass == "blocking" and breaks not in ids:
            disclosures.append(f"blocking_without_valid_breaks:{fid}")
            klass = "note"
        if klass == "need_evidence":
            if not locator:
                disclosures.append(f"need_evidence_without_locator:{fid}")
                klass = "note"
            elif len(locator) > MAX_ITEM_CHARS:
                # W3 host attachment is bounded like the agent's own evidence items: an over-long
                # locator is never remembered (state) nor attached — demoted, disclosed.
                disclosures.append(f"need_evidence_locator_too_long:{fid}")
                klass = "note"
            elif locator not in seen and len(seen) >= MAX_NEED_EVIDENCE_MEMORY:
                disclosures.append(f"need_evidence_memory_full:{fid}")
                klass = "note"
            elif locator in seen:
                # I-03: a repeat is DEMOTED, never dropped. Dropping it removed the finding from
                # the aggregate, so a reviewer re-asking for evidence it still needs turned the
                # wave GREEN and closed the gate. Demotion keeps the cost bound (never attached
                # again, never blocking, no new fingerprint) while the wave stays open until the
                # agent disposes of it.
                disclosures.append(f"need_evidence_repeat:{locator}")
                klass = "note"
            else:
                seen.add(locator)
        normalized.append({
            "id": fid, "class": klass, "breaks": breaks,
            # a stored locator is bounded like every other finding string (visible marker)
            "locator": bounded_text(locator, MAX_FINDING_TEXT_CHARS),
            "summary": summary,
            "recommendation": bounded_text(item.get("recommendation"), MAX_FINDING_TEXT_CHARS),
        })
    return normalized, disclosures, frozenset(seen)


def aggregate(slot_results: Iterable[Mapping[str, Any]], *, quorum: Optional[int] = None) -> dict:
    """Host-computed wave aggregate — no reviewer-authored verdict.

    ``slot_results`` = EVERY configured slot as ``{slot, model, ok, findings,
    error?}`` (``ok`` = parseable; ``findings`` = ``validate_findings`` output).
    ``quorum`` defaults to ``config.adaptive_quorum(len(slot_results))``.
    parseable slots < quorum → ``DEGRADED``; slots with ≥1 ``blocking`` ≥ quorum
    → ``REVISE_PLAN``; any other finding → ``REVIEW_REQUIRED`` (a
    need_evidence-only wave never revises); nothing → ``GREEN``. One configured
    slot follows ``adaptive_quorum(1) == 1`` (its blocking finding revises) with
    the loud reason ``single_reviewer_no_diversity``. Returns
    ``{aggregate, reasons, counts, findings}`` — findings flattened with
    ``finding_id = "<slot>:<id>"``, ``slot`` and ``model`` stamped.
    """
    rows = [dict(row) for row in slot_results]
    configured = len(rows)
    q = max(1, int(quorum)) if quorum is not None else adaptive_quorum(configured)
    reasons: list[str] = []
    flat: list[dict] = []
    blocking_slots = 0
    parseable = 0
    labels: set[str] = set()
    for position, row in enumerate(rows, start=1):
        slot = str(row.get("slot") or "") or str(position)
        if slot in labels:
            slot = f"{slot}_{position}"  # duplicate labels must not merge finding_ids
        labels.add(slot)
        if not row.get("ok"):
            reasons.append(f"slot_unparseable:{slot}:{str(row.get('error') or 'no parseable findings')}")
            continue
        parseable += 1
        slot_findings = [dict(f) for f in (row.get("findings") or []) if isinstance(f, Mapping)]
        if any(f.get("class") == "blocking" for f in slot_findings):
            blocking_slots += 1
        for finding in slot_findings:
            finding.update({
                "finding_id": f"{slot}:{finding.get('id', '')}", "slot": slot,
                "model": str(row.get("model") or ""),
            })
            flat.append(finding)
    counts = {
        "configured": configured, "parseable": parseable, "quorum": q,
        "blocking_slots": blocking_slots,
        **{klass: sum(1 for f in flat if f.get("class") == klass) for klass in FINDING_CLASSES},
    }
    if configured == 1:
        reasons.append("single_reviewer_no_diversity")
    if configured == 0 or parseable < q:
        verdict = "DEGRADED"
        reasons.append(f"parseable_slots_below_quorum:{parseable}/{q}")
    elif blocking_slots >= q:
        verdict = "REVISE_PLAN"
        reasons.append(f"blocking_slots_at_quorum:{blocking_slots}/{q}")
    elif flat:
        verdict = "REVIEW_REQUIRED"
        if counts["blocking"]:
            reasons.append(f"blocking_below_quorum:{blocking_slots}/{q}")
        if counts["need_evidence"] == len(flat):
            reasons.append("need_evidence_only")
    else:
        verdict = "GREEN"
    return {"aggregate": verdict, "reasons": reasons, "counts": counts, "findings": flat}


def blocking_fully_rejected(findings, dispositions) -> bool:
    """Whether EVERY blocking finding carries exactly one VALID reject disposition.

    The earned-delta admission (a fully-rejected REVISE_PLAN wave buys its promised
    delta cycle) must not trust raw items: an empty rationale, an unknown id, or a
    contradictory accept+reject pair is refused by the closure table and must not
    earn a paid panel (delta-review finding D1)."""
    blocking = [f for f in findings or [] if isinstance(f, Mapping) and f.get("class") == "blocking"]
    if not blocking:
        return False
    known = {str(f.get("finding_id") or f.get("id") or "") for f in findings or [] if isinstance(f, Mapping)}
    seen: dict[str, str] = {}
    invalid: set[str] = set()
    for item in dispositions or []:
        if not isinstance(item, Mapping):
            continue
        fid = str(item.get("finding_id") or "").strip()
        decision = str(item.get("decision") or "").strip().lower()
        ok = decision in DISPOSITION_DECISIONS and bool(str(item.get("rationale") or "").strip())
        if fid not in known:
            return False  # D3: an unknown id makes the whole disposition invalid — no earned cycle
        if not ok or fid in seen:
            invalid.add(fid)
            continue
        seen[fid] = decision
    return all(
        (fid := str(f.get("finding_id") or f.get("id") or "")) not in invalid
        and seen.get(fid) == "reject"
        for f in blocking
    )


def closure_after_disposition(
    aggregate: str,
    findings: Iterable[Mapping[str, Any]],
    dispositions: Iterable[Mapping[str, Any]],
    enforcement: str,
) -> dict:
    """The ONE closure table (F7) → ``{closed, open_ids, notes}``.

    GREEN → closed. REVIEW_REQUIRED (only note/need_evidence) → closed when
    every finding id carries a disposition (accept|reject|defer + rationale —
    the disposition form as today, plan §7.2 A). REVISE_PLAN → NEVER closed by
    disposition (blocking needs a changed spec → new cycle, or reject-with-
    rationale → next paid delta cycle). DEGRADED → not closable by disposition
    (rerun the wave). Advisory enforcement never flips ``closed``: the caller
    may proceed with the wave open under loud disclosure — this function only
    reports. Control-line invariants (``plan_render
    ._parse_plan_review_control``): GREEN ⇒ closed; REVISE_PLAN and DEGRADED ⇒
    not closed (B2 — DEGRADED reaches the control line as itself and is always
    OPEN, instead of being laundered into REVIEW_REQUIRED).
    """
    verdict = str(aggregate or "").strip().upper()
    mode = str(enforcement or "").strip().lower()
    items = [dict(f) for f in findings if isinstance(f, Mapping)]
    valid: dict[str, dict] = {}
    duplicates: set[str] = set()
    notes: list[str] = []
    for item in dispositions or []:
        if not isinstance(item, Mapping):
            continue
        fid = str(item.get("finding_id") or "").strip()
        decision = str(item.get("decision") or "").strip().lower()
        if decision not in DISPOSITION_DECISIONS or not str(item.get("rationale") or "").strip():
            notes.append(f"invalid_disposition:{fid or '<no finding_id>'}")
            continue
        if fid in valid:
            # Two entries for one finding are contradictory intent, not an update: silently
            # keeping the last would let "accept, then reject" close a wave on whichever the
            # agent happened to write second. Both are refused; the finding stays open.
            notes.append(f"duplicate_disposition:{fid}")
            valid.pop(fid, None)
            duplicates.add(fid)
            continue
        if fid in duplicates:
            continue
        valid[fid] = dict(item)
    known = {str(f.get("finding_id") or f.get("id") or "") for f in items}
    notes.extend(f"unknown_finding_id:{fid}" for fid in sorted(set(valid) - known))
    open_ids: list[str] = []
    for finding in items:
        fid = str(finding.get("finding_id") or finding.get("id") or "")
        blocking = finding.get("class") == "blocking"
        # C-08: a validated BLOCKING finding stays open whatever the aggregate label
        # says — a single blocking finding below quorum surfaces as REVIEW_REQUIRED,
        # and closing it with a $0 disposition would be exactly the laundering the
        # height rule exists to prevent. It needs a changed spec or a paid delta cycle.
        if blocking or fid not in valid:
            open_ids.append(fid)
    if verdict == "GREEN":
        closed = True
    elif verdict == "REVIEW_REQUIRED":
        closed = not open_ids
        if not items:
            notes.append("no_findings_recorded: REVIEW_REQUIRED without findings closes vacuously")
        if any(f.get("class") == "blocking" for f in items):
            notes.append(
                "blocking_finding_below_quorum_stays_open: revise the spec or let the next "
                "paid delta cycle judge the rejection"
            )
    elif verdict == "REVISE_PLAN":
        closed = False
        notes.append(
            "revise_plan_not_closable_by_disposition: blocking findings need a changed spec "
            "(new cycle) or reject-with-rationale judged in the next paid delta cycle"
        )
    elif verdict == "DEGRADED":
        closed = False
        notes.append("degraded_not_closable_by_disposition: fewer parseable reviewer slots than quorum — rerun the wave")
    else:
        closed = False
        notes.append(f"unknown_aggregate:{verdict or '<empty>'}")
    if not closed:
        notes.append(
            "advisory_enforcement: the caller may proceed with the wave open under loud disclosure"
            if mode == "advisory" else
            "blocking_enforcement: the wave must close before the work starts"
        )
    return {"closed": closed, "open_ids": open_ids, "notes": notes}
