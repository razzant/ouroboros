"""Deep self-review of the whole Ouroboros system against BIBLE.md.

``/review`` is ``review_change(subject=system, surface=system)``
(``tools/review_change.run_system_review``): the review runs on the ONE row the
call names — any enabled catalog row — or, by default, the direct Main row
(``main_review_row``), and its report is that record's answer. Every row
DELIVERS BY RETRIEVAL — the reviewer reads the repository itself — in one of
two shapes:

* an ``api_chat`` row (a bare route or a configured-subagent reference) is a
  NATIVE inspection episode — the reviewer reads the repository through the
  host's read-only tools (the runtime root is its readable data plane) while
  BIBLE.md, the standing disclosures and memory whitelist reach it inline;
  repository reads and inline delivery remain distinct coverage evidence;
* an ``agent_session`` row is a delegated read-only session — the same task,
  reads not host-observed (disclosed as ``unobserved``).

Both deliveries ride the shared review executor seam: the product
is free markdown (``triad_review`` shape ``report``), a bound landing before
the final answer delivers the collected draft marked INCOMPLETE, and the host
prepends a provenance header naming the delivery, model, rounds, receipts,
coverage and completeness so consecutive reports stay comparable.
"""

from __future__ import annotations

import logging
import json
import pathlib
import posixpath
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

log = logging.getLogger(__name__)

from ouroboros.tools.review_helpers import (  # noqa: E402
    _MAX_FULL_REPO_FILE_BYTES,
    author_questions_block,
    load_governance_doc,
)
from ouroboros.shell_parse import is_absolute_path_text  # noqa: E402
from ouroboros.utils import utc_now_iso  # noqa: E402
from ouroboros.provider_models import provider_for_model, provider_has_credentials  # noqa: E402
from ouroboros.reviewer_slot_config import (  # noqa: E402
    ROUTE_KIND_API,
    ROUTE_KIND_SESSION,
    ConfiguredReviewerSlot,
    row_effort,
)
from ouroboros.usage_accounting import BudgetExceeded  # noqa: E402
from ouroboros.triad_review import REVIEW_REPORT_CONTRACT  # noqa: E402
from ouroboros.config import runtime_setting

# The report's own output reserve: what the row's model may spend answering,
# on the request and on the slot (one number for both deliveries).
_DEEP_MAX_OUTPUT_TOKENS = 100_000

_MEMORY_WHITELIST = [
    "memory/identity.md",
    "memory/scratchpad.md",
    "memory/registry.md",
    "memory/WORLD.md",
    "memory/knowledge/index-full.md",
    "memory/knowledge/patterns.md",
    "memory/knowledge/improvement-backlog.md",
]

# The role half of the reviewer prompt, shared by both deliveries; the "how to
# work" half below tells the reviewer which tools it reads the repository with.
_ROLE_PROMPT = """\
You are conducting a deep self-review of the Ouroboros project — a self-creating AI agent.

Primary directive: The Constitution (BIBLE.md) is your absolute reference.
Every finding must be checked against it.

What to look for: bugs, crashes, race conditions,
BIBLE.md violations (P0–P12), contradictions between code and docs,
security gaps, dead code, missing error handling, architectural issues,
known error patterns from patterns.md that remain unfixed, and ideas how to improve Ouroboros to work better and better comply with the Bible."""

# The repository is inspected with tools; tier-1 governance and memory arrive
# inline. No second tool read is needed for those exact delivered documents.
_RETRIEVING_METHOD = """

How to work: inspect the repository yourself with read-only tools. `BIBLE.md`
({bible_chars:,} chars) and the standing disclosures are delivered IN FULL below:
use them as your reference without a redundant tool read. Every finding is checked
against the constitution. The governance sections that follow are the rules delivered to you in
full; the governance navigation names every other document, `docs/ARCHITECTURE.md`
and `docs/DEVELOPMENT.md` included, as book entrypoints whose overviews identify the
actual physical chapter sources. The memory files below are inlined byte-exact. Read
the needed sources on demand, using lines local to the physical file you open,
never treating composed-book line numbers as an entrypoint address. Then inspect the code (search_code, query_code,
read_file), cross-reference interactions between modules and follow call chains out
of the files you open. Prioritize: CRITICAL > IMPORTANT > ADVISORY.

Output: Structured markdown report, MOST CRITICAL findings first, each citing the
specific file, line/section, the problem, and the proposed fix. Begin with a one-line
coverage header naming what you actually read (documents and files, in full or by
section) and what you did not — your host records only the reads it observed."""

# The report contract for a retrieving row: the shared report shape plus the
# deep review's own coverage-header sentence. The SHAPE is already `report`
# through REVIEW_OUTPUT_SHAPES (the executors' default contract for it is the
# report contract, never an array); the policy hands over the same contract
# WITH the deep-review sentence, so the ask and the parse cannot disagree.
_REPORT_CONTRACT = REVIEW_REPORT_CONTRACT + (
    "Begin with one line naming what you read (in full or by section) and what you did "
    "not; the rest is the prioritized report."
)

_MANDATORY_READS = ("BIBLE.md",)
# The inspection roots that resolve to the REPOSITORY (both name the review's
# session root); a read under the data plane never satisfies a repository read.
_REPO_ROOTS = frozenset({"", "active_workspace", "system_repo"})

# The standing goal of a bare ``/review``: the one spelling the request, the ledger
# record and the reviewer's brief share.
STANDING_GOAL = "Deep self-review of the whole Ouroboros system against BIBLE.md."


@dataclass(frozen=True)
class SystemReviewAsk:
    """What the caller asks of this self-review BESIDE its standing questionnaire —
    ``review_change(subject=system, surface=system)``'s ``goal``, ``author_questions``
    and ``reason``. ONE typed object reaches the brief builder and the ledger record,
    so what the reviewer is asked and what the record says it was asked cannot
    diverge: an empty goal is the standing goal; the questions are put to the
    reviewer verbatim, after its own questionnaire; the reason names why this row
    was chosen and is recorded with the panel."""

    goal: str = ""
    author_questions: Tuple[str, ...] = ()
    reason: str = ""

    @property
    def effective_goal(self) -> str:
        return self.goal.strip() or STANDING_GOAL

    def brief_section(self) -> str:
        """The caller's addition to the task text: its goal when it set one, then its
        questions numbered as asked; ``""`` when the call asked nothing of its own."""
        parts = []
        if self.goal.strip():
            parts.append(f"The caller's goal for this review (beside the standing review above): {self.goal.strip()}")
        questions = author_questions_block(self.author_questions, note="after your own questionnaire")
        if questions:
            parts.append(questions)
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Availability — route-aware on the configured row.
# ---------------------------------------------------------------------------


def _api_route_model(row: ConfiguredReviewerSlot) -> Tuple[str, Optional[str]]:
    """The api row's ``(unavailable_reason, sendable_model)``.

    Credential knowledge is the provider registry's SSOT. The ONE deep-review
    rule that lives here is the direct-OpenAI resolution: an install whose only
    OpenAI access is the direct API (no OpenRouter key, ``OPENAI_BASE_URL``
    unset) runs a stored OpenRouter spelling ``openai/<slug>`` on the direct
    ``openai::<slug>`` route, so a row saved on an OpenRouter install keeps
    working here. A ``-pro`` suffix is an OpenRouter ROUTING slug (reasoning
    mode), not an OpenAI model id — ``gpt-5.6-sol-pro`` 404s on api.openai.com
    (live-probed 2026-07-29) — so those land on the direct route's own
    deep-review default, while an explicit pin of a REAL model keeps the
    mechanical rewrite (a pinned ``openai/gpt-5.5`` runs on ``gpt-5.5``). The
    resolved spelling is what the review is actually sent on; the row's own
    effort, credential pin and local-route flag are untouched.
    """
    from ouroboros.provider_models import model_has_credentials

    configured = str(row.target_id)
    if row.use_local is True or model_has_credentials(configured):
        return "", configured
    if configured.startswith("openai/"):
        if provider_has_credentials("openai") and not runtime_setting("OPENAI_BASE_URL"):
            from ouroboros.provider_models import OPENAI_DIRECT_DEFAULTS

            slug = configured.split("/", 1)[1]
            return "", (OPENAI_DIRECT_DEFAULTS["deep_self_review"] if slug.endswith("-pro")
                        else "openai::" + slug)
        return f"no OpenRouter or direct OpenAI credentials for {configured}", None
    return f"no {provider_for_model(configured)} credentials for {configured}", None


def _session_route_reason(row: ConfiguredReviewerSlot) -> str:
    """Why a delegated session row cannot run now, or '' — the substrate's own
    route health (the executor refuses on the same reader before it starts)."""
    from ouroboros.subagents import delegated_run_shape, parse_subagent_harness, route_health

    route = parse_subagent_harness(row.session_target or row.target_id)
    if route is None:
        return "session_target_unparsable"
    try:
        from ouroboros.claudexor_daemon import ensure_owned_gateway

        gateway = ensure_owned_gateway(admission_wait_sec=0)
    except Exception as exc:
        return f"agent_service_unavailable: {type(exc).__name__}: {exc}"
    try:
        unavailable, _reset_at = route_health(
            gateway, route.route_id, delegated_run_shape(False), route_model=route.model,
            pinned_profile=str(row.profile_id or getattr(route, "profile_id", "") or ""),
        )
    finally:
        gateway.close()
    return str(unavailable or "")


def main_review_row() -> ConfiguredReviewerSlot:
    """The direct Main row — ``OUROBOROS_MODEL``, its local flag and Main's pinned
    account (``OUROBOROS_MODEL_ACCOUNTS["main"]``): ``/review``'s executor when the
    call names none (decision 3A) and the deep self-review default.

    The pin rides as the row's credential profile because the executor sends under
    ``model_role=reviewer:main`` with ``model_account_override=<row profile>``, and an
    EMPTY override is Auto, never a lookup of Main's own role: without it a pinned
    Main would review on whatever account the engine picked. An unpinned Main stays
    Auto; an invalid pin document raises here as it does for Main's own calls.
    """
    from ouroboros.model_slots import MODEL_ACCOUNTS_KEY, model_role_option
    from ouroboros.subagents import _lane_model, _use_local_for_lane

    model = _lane_model("main")
    return ConfiguredReviewerSlot(slot_id="main", kind=ROUTE_KIND_API, target_id=model,
                                  use_local=True if _use_local_for_lane("main", model) else None,
                                  profile_id=str(model_role_option(MODEL_ACCOUNTS_KEY, "main") or ""))


def deep_review_route(row: Optional[ConfiguredReviewerSlot] = None) -> Tuple[str, Optional[str]]:
    """``(unavailable_reason, identity)`` for the deep-review row (default: Main).

    '' means available; ``identity`` is then what the review runs on — the api
    row's sendable model (``_api_route_model``, the direct-OpenAI resolution
    included) or the session row's ``harness[=model]`` target. Availability is
    ROUTE-AWARE: an api row needs its routed model's credentials, a session row
    a healthy delegated route.
    """
    row = row or main_review_row()
    if row.kind not in (ROUTE_KIND_API, ROUTE_KIND_SESSION):
        return f"deep_review row has an unknown route kind {row.kind!r}", None
    if not str(row.target_id or "").strip():
        return "deep_review row has no target (empty model id / session target)", None
    if row.is_session:
        reason = _session_route_reason(row)
        return reason, (None if reason else (row.session_target or row.target_id))
    return _api_route_model(row)


def deep_review_unavailable_text(reason: str) -> str:
    """The ONE unavailable message (prefix classified by ``outcomes``)."""
    return (
        f"❌ Deep self-review unavailable: {reason}. Run /review on a row this install can pay: "
        "any enabled Settings → Agents row, or the Main model (the default)."
    )


# ---------------------------------------------------------------------------
# The two retrieving deliveries.
# ---------------------------------------------------------------------------


def _record_execution(slot: Any, usage: Dict[str, Any], *, status: str, error: str = "") -> None:
    """«Выполняется как» (D22) for the deep-review row — disclosure, best-effort."""
    try:
        from ouroboros.review_substrate import ReviewActorRecord
        from ouroboros.reviewer_slot_config import record_reviewer_slot_executions

        actor = ReviewActorRecord(slot_id=slot.slot_id, model=slot.model, status=status,
                                  usage=dict(usage or {}), error=error)
        record_reviewer_slot_executions("deep_self_review", [actor], {slot.slot_id: slot})
    except Exception:
        log.debug("deep self-review last-execution write failed", exc_info=True)


def _repo_relative(path: Any, repo_dir: pathlib.Path) -> str:
    """A receipt path as a repo-relative POSIX path — on EVERY host OS.
    Coverage hands it the receipt's ``opened_path`` (the root-relative path
    the reader actually opened — already free of the model's spelling:
    absolute, whitespace-padded, ``repo/``-prefixed or ``/``-qualified forms
    all arrive as ``BIBLE.md``) and, for a receipt WITHOUT one (nothing
    rendered), the raw spelling. Absolute paths under the repository are
    relativized, relative ones normalized — but a ``..`` component is kept AS
    SPELLED and so names no mandatory read: the registry refuses traversal
    shapes before dispatch, so ``a/../BIBLE.md`` delivered nothing and is
    never folded onto ``BIBLE.md``. The POSIX contract is by construction:
    separators are folded to ``/`` first, absolute spellings are recognized
    for every OS (``/``, drive-letter and UNC forms — ``PurePosixPath`` alone
    is blind to ``C:/``), and normalization is ``posixpath``'s, never
    ``os.path``'s, whose Windows form renders ``docs\\ARCHITECTURE.md`` and
    would never match a mandatory read."""
    text = str(path or "").replace("\\", "/")
    pure = pathlib.PurePosixPath(text)
    if ".." in pure.parts:
        return text
    if is_absolute_path_text(text):
        try:
            return pathlib.Path(text).resolve().relative_to(pathlib.Path(repo_dir).resolve()).as_posix()
        except (ValueError, OSError):
            return pure.as_posix()
    return posixpath.normpath(text).removeprefix("./")


def _native_read_coverage(usage: Dict[str, Any], repo_dir: pathlib.Path) -> Dict[str, Dict[str, Any]]:
    """Prefer the native operation's exact manifest-bound source coverage.

    Historical receipts retain only their explicitly labelled line evidence.
    R8: how much of each mandatory read the host OBSERVED, from the episode's
    receipts — the merged line intervals of every executed repository-root
    ``read_file`` receipt for the path (a single result is capped, so a full
    read of BIBLE.md is multi-chunk by construction).

    ``read`` only when the union of extent-bearing receipts covers the whole
    file; otherwise ``unobserved`` when the receipt list was capped below the
    call count OR any matching executed receipt carries no extent (absence
    proves nothing there — full coverage must be proven by measured receipts
    alone); ``partial`` with the covered fraction; ``missing`` when nothing of
    the file was delivered — no receipt names it at a repository root (a
    data-plane read never counts), or every measured receipt delivered zero
    lines (a cursor past the window, a start past EOF). Receipts are matched
    on the path AND root the reader actually OPENED (``opened_path`` /
    ``opened_root``, stamped by the reader; the model's spellings are only
    disclosure — a padded ``" system_repo "`` counts, a ``runtime_data`` read
    never does), falling back to the raw spellings for a receipt that rendered
    nothing — where a ``..`` component names nothing (the registry refuses
    traversal shapes before dispatch — see ``_repo_relative``). Disclosure,
    never a refusal: the report is delivered with the flag in its header.
    """
    exact = usage.get("native_read_coverage")
    sources = exact.get("sources") if isinstance(exact, dict) else None
    if isinstance(sources, list) and sources:
        out = {}
        for index, source in enumerate(sources):
            source = source if isinstance(source, dict) else {}
            root, path = str(source.get("root") or ""), str(source.get("path") or f"source[{index}]")
            address = path if root in _REPO_ROOTS else f"{root}:{path}"
            if address in out:
                address = f"{root}:{path}@{source.get('source_revision', index)}"
            total, covered = source.get("complete_chars"), source.get("covered_chars")
            known = (type(total) is int and type(covered) is int and 0 <= covered <= total
                     and isinstance(source.get("source_revision"), str) and len(source["source_revision"]) == 64)
            state = ("delivered_inline" if source.get("status") == "complete" and known
                     and source.get("coverage_basis") == "delivered_inline" else
                     "read" if source.get("status") == "complete" and known else
                     "partial" if source.get("status") == "incomplete" and known and covered else
                     "missing" if source.get("status") == "incomplete" and known else "unobserved")
            out[address] = {**source, "state": state, "covered_chars": covered if known else 0,
                            "complete_chars": total if known else 0,
                            "fraction": round(covered / total, 3) if known and total else 1.0 if state == "read" else 0.0,
                            "evidence_basis": "source_ranges"}
        return out
    receipts = [r for r in (usage.get("native_tool_receipts") or []) if isinstance(r, dict)]
    capped = int(usage.get("native_tool_calls") or 0) > len(receipts)
    out: Dict[str, Dict[str, Any]] = {}
    for rel in _MANDATORY_READS:
        spans: list[tuple[int, int]] = []
        total, unmeasured = 0, False
        for r in receipts:
            named = r.get("opened_path") if isinstance(r.get("opened_path"), str) and r.get("opened_path") else r.get("path")
            root = r.get("opened_root") if isinstance(r.get("opened_root"), str) and r.get("opened_root") else str(r.get("root") or "")
            if (r.get("tool") != "read_file" or r.get("outcome") != "executed"
                    or r.get("delivered") is False
                    or root not in _REPO_ROOTS or _repo_relative(named, repo_dir) != rel):
                continue
            if not all(isinstance(r.get(k), int) for k in ("start_line", "end_line", "total_lines")):
                # Names the file but carries no extent: it may never have opened
                # it (an argument error, a registry refusal answered with text)
                # or opened it without a recorded extent — either way it proves
                # nothing, and keeps `read`/`missing` unproven (`unobserved`).
                unmeasured = True
                continue
            total = max(total, int(r["total_lines"]))
            if r["end_line"] >= r["start_line"]:
                spans.append((int(r["start_line"]), int(r["end_line"])))
        covered, cursor = 0, 0
        for start, end in sorted(spans):  # merge overlapping / re-read chunks; clip to the file
            lo, hi = max(start, cursor + 1, 1), min(end, total)
            if hi >= lo:
                covered += hi - lo + 1
                cursor = hi
        if total and covered >= total:
            state = "read"
        elif capped or unmeasured:
            state = "unobserved"
        else:
            state = "partial" if covered else "missing"
        out[rel] = {"state": state, "covered_lines": covered, "total_lines": total,
                    "fraction": round(covered / total, 3) if total else 0.0,
                    "evidence_basis": "legacy_lines"}
    return out


_HEADER_VALUE_MAX_CHARS = 120


def _header_value(value: Any) -> str:
    """One header value, bounded and unable to break the comment or the line:
    newlines become spaces, `--` (the comment terminator's body) collapses to
    `-`, and the text is cut disclosed at a fixed bound."""
    from ouroboros.utils import truncate_within_limit

    text = truncate_within_limit(str(value), _HEADER_VALUE_MAX_CHARS)
    # Sanitize AFTER the bound: the disclosed omission marker itself carries a
    # newline, and nothing may leave this function able to break the comment.
    text = text.replace("\r", " ").replace("\n", " ")
    while "--" in text:
        text = text.replace("--", "-")
    return text


def _delivery_incomplete(delivery: str, usage: Dict[str, Any]) -> str:
    """An interrupted report remains incomplete; measured reading coverage is
    a separate diagnostic and cannot make a finished report unfinished."""
    if delivery == "native_tool_rounds":
        reported = str(usage.get("native_incomplete") or "")
        if reported == "required_source_coverage_incomplete":
            reported = ""
        return reported or "none"
    return "unobserved"


def _provenance_header(delivery: str, model: str, usage: Dict[str, Any], memory: Dict[str, Any],
                       coverage: Dict[str, str], human: str, *, incomplete: str,
                       extra: Optional[Dict[str, Any]] = None) -> str:
    """R9: the host's provenance header (machine-readable comment + one human
    line) prepended to every delivered report. The fact set is built PER
    DELIVERY (a session never carries rounds/receipts; its attestation is
    `unobserved` by construction); every comment value goes through
    `_header_value` (bounded, sanitized), and the human line — whose external
    values the callers pass through `_header_value` too — is kept to one line
    with no comment terminator in it."""
    facts: Dict[str, Any] = {"delivery": delivery, "model": model, "memory": f"{memory['inlined']}/{memory['total']}"}
    # One value PER disposition (`memory_missing=…`, `memory_empty=…`, …): each
    # lists at most the seven whitelisted basenames (≈91 chars worst case), so
    # it fits the value bound; `_header_value` bounds it regardless.
    for d in ("missing", "empty", "oversized", "read_error"):
        names = [rel.rsplit("/", 1)[-1] for rel, got in memory["dispositions"].items() if got == d]
        if names:
            facts[f"memory_{d}"] = ",".join(names)
    facts["coverage"] = ",".join(f"{rel}:{state}" for rel, state in coverage.items())
    facts["incomplete"] = incomplete  # computed ONCE by the caller from the facts its delivery holds
    if delivery == "native_tool_rounds":
        facts.update({
            "attestation": usage.get("host_file_read_attestation") or "unobserved",
            "rounds": usage.get("native_rounds", 0), "tool_calls": usage.get("native_tool_calls", 0),
            "receipts": len(usage.get("native_tool_receipts") or []),
            "end_reason": usage.get("native_end_reason", ""),
            "transcript": f"{usage.get('native_transcript_chars', 0)}/{usage.get('native_transcript_bound', 0)}",
            "landing": f"{usage.get('native_landing_notified', False)}/{usage.get('native_landing_sent', False)}",
            "coverage_basis": usage.get("deep_review_coverage_basis", "legacy_lines"),
        })
    else:
        facts["attestation"] = "unobserved"
    facts.update(extra or {})
    # Report generation is observable; none of these delivery paths freezes one
    # reviewed Git revision. Do not relabel a live HEAD or file mtime as that proof.
    generated_at = utc_now_iso()
    usage["deep_review_generated_at"] = generated_at
    facts.update({"generated_at": generated_at, "source_revision": "unknown"})
    comment = ", ".join(f"{key}={_header_value(value)}" for key, value in facts.items())
    line = str(human).replace("\r", " ").replace("\n", " ")
    while "--" in line:  # the callers bound each external value; the line itself never carries a terminator
        line = line.replace("--", "-")
    return (
        f"<!-- deep-review provenance: {comment} -->\n_{line}_\n"
        f"Report generated at {generated_at}; reviewed source revision: unknown (not captured).\n\n"
    )


def _memory_line(memory: Dict[str, Any]) -> str:
    """The human half of the memory fact: `memory 3/7 inlined (omitted: …)`."""
    omitted = [f"{rel.rsplit('/', 1)[-1]} {d}" for rel, d in memory["dispositions"].items() if d != "inlined"]
    return f"memory {memory['inlined']}/{memory['total']} inlined" + (f" (omitted: {', '.join(omitted)})" if omitted else "")


def _failed(text: str, *, reason_code: str, usage: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
    """A failure result: the text plus TYPED usage, so the caller keeps the
    previous report instead of overwriting durable memory with an error.
    Callers spell ``reason_code`` as a literal — the runtime's reason-code
    drift guard (outcomes vocabulary) reads emit sites, not constants."""
    out = dict(usage or {})
    out.update({"execution_status": "infra_failed", "reason_code": reason_code})
    return text, out


def _retrieving_task(repo_dir: pathlib.Path, drive_root: pathlib.Path, *,
                     usable_window_tokens: int = 0,
                     required_sources: Optional[list] = None,
                     required_sources_ref: Optional[dict] = None,
                     ask: Optional[SystemReviewAsk] = None) -> Tuple[str, Dict[str, Any]]:
    """The route-owned task text for a deep-review row: role + method, the caller's
    own goal and questions when it set any (``ask``, after the standing questionnaire),
    the governance tiers this surface receives, and the memory whitelist inline
    byte-exact.

    The tiers come from the ONE SSOT every review surface asks
    (``governance_context``): the standing disclosures and the review protocol
    arrive in full, the reference books arrive as navigation this reviewer
    reads with its own tools, and every document that is not inlined is named
    in the navigation. ``usable_window_tokens`` is the window the inline share
    is taken against (0 asks for navigation for tiers 2 and 3 only). Tier 1 is
    always inline. Exact matching required sources record that delivery, so
    the reviewer need not reread a document it already received in full."""
    from ouroboros.tools.governance_context import governance_context
    from ouroboros.tools.scope_required_sources import source_text_identity, with_inline_sources

    bible = load_governance_doc(repo_dir, "BIBLE.md", on_missing="silent")
    if not bible.strip():
        raise RuntimeError("BIBLE.md is missing at the repository root — a deep self-review has no constitution to check against")
    # The memory whitelist, inlined byte-exact, and the typed memory fact every
    # delivery carries: ``{"inlined": n, "total": 7, "dispositions": {rel:
    # inlined | missing | empty | oversized | read_error}}`` — one disposition
    # per whitelisted path (task text, usage fact, provenance header), never a
    # silent skip.
    memory_parts: list[str] = []
    dispositions: Dict[str, str] = {}
    for rel_mem in _MEMORY_WHITELIST:
        full_path = drive_root / rel_mem
        try:
            if not full_path.is_file():
                dispositions[rel_mem] = "missing"
                continue
            if full_path.stat().st_size > _MAX_FULL_REPO_FILE_BYTES:
                dispositions[rel_mem] = "oversized"
                continue
            content = full_path.read_text(encoding="utf-8", errors="replace")
            if not content.strip():
                dispositions[rel_mem] = "empty"
                continue
            memory_parts.append(f"## FILE: drive/{rel_mem}\n{content}\n")
            dispositions[rel_mem] = "inlined"
        except Exception:
            dispositions[rel_mem] = "read_error"
            log.debug("memory whitelist entry unreadable: %s", rel_mem, exc_info=True)
    memory = {"inlined": sum(1 for d in dispositions.values() if d == "inlined"),
              "total": len(_MEMORY_WHITELIST), "dispositions": dispositions}
    governance = governance_context(
        repo_dir, surface="deep_self_review", touched_paths=(),
        usable_window_tokens=usable_window_tokens, delivery="retrieving",
        checklist_section_text="")
    sources = required_sources if required_sources is not None else [
        {"root": "system_repo", "path": path,
         **source_text_identity(governance.inline_whole_documents[path].encode("utf-8"))}
        for path in _MANDATORY_READS]
    sources = with_inline_sources(sources, governance.inline_whole_documents)
    parts = [
        _ROLE_PROMPT + _RETRIEVING_METHOD.format(bible_chars=len(bible)),
        (ask or SystemReviewAsk()).brief_section(),
        governance.stable_inline,
        governance.selected_inline,
        governance.navigation,
        "## Memory (runtime data root, inlined byte-exact)",
        *memory_parts,
        # EVERY whitelisted entry gets its disposition here — this delivery's
        # omission disclosure — so an absent or blank memory file is a stated
        # fact the reviewer (and the header) can rely on, never a gap.
        f"Memory dispositions ({memory['total']} whitelisted): "
        + "; ".join(f"{rel} {d}" for rel, d in memory["dispositions"].items()),
    ]
    if required_sources_ref:
        parts.append("Read the complete required source manifest through its exact source handle and inspect its sources across your working views. Documents already delivered in full above need no second tool read: "
                     + json.dumps(required_sources_ref, ensure_ascii=False))
    # The delivered tiers need no second record here: the whole task text —
    # its inline governance and the navigation that names every document it
    # did not inline — is persisted with the request in prompt custody.
    return "\n\n".join(part for part in parts if str(part or "").strip()), {
        "memory": memory, "bible_chars": len(bible), "required_sources": sources,
        "governance_manifest": governance.manifest}


def _review_usage_scope(current: Any, request: Any) -> Any:
    """The review's own usage scope: ``source`` names the surface and the wave this review's
    round (#1544); the CATEGORY stays the tree's when that tree is one consciousness started
    (``consciousness``/``consciousness_task``), because the rolling allowance discovers its roots
    by that category — a review root whose only priced rows said ``deep_self_review`` was
    invisible to it (review round 3)."""
    from dataclasses import replace

    from ouroboros.consciousness_allowance import CONSCIOUSNESS_CATEGORIES
    from ouroboros.review_records import resolve_review_wave

    category = str(getattr(current, "category", "") or "")
    keep = category in CONSCIOUSNESS_CATEGORIES
    wave = resolve_review_wave(request, request.usage_attribution, str(getattr(current, "review_wave_id", "") or ""))
    return replace(current, category=category if keep else "deep_self_review", source="deep_self_review",
                   review_wave_id=wave)


def _run_retrieving_review(
    repo_dir: pathlib.Path,
    drive_root: pathlib.Path,
    llm: Any,
    emit_progress: Callable[[str], None],
    row: ConfiguredReviewerSlot,
    *,
    task_id: str,
    deadline_at: str,
    model: str = "",
    required_sources: Optional[list] = None,
    required_sources_ref: Optional[dict] = None,
    ask: Optional[SystemReviewAsk] = None,
) -> Tuple[str, Dict[str, Any]]:
    """The row's delivery (native episode or delegated session): hand-built
    request, slot and assignment; the product is the
    report text. ``model`` is the sendable spelling ``deep_review_route``
    resolved for the row (its own target when the caller names none); ``ask`` is
    the caller's goal and questions, in the request's goal and the task text."""
    from dataclasses import asdict

    from ouroboros.config import get_finalization_grace_sec, get_task_abs_ceiling_sec, operation_window_sec
    from ouroboros.deadline_utils import review_operation_timeout_sec
    from ouroboros.observability import persist_call
    from ouroboros.review_execution import ReviewAssignment, _review_route_executor
    from ouroboros.review_native_episode import review_native_transcript_bound
    from ouroboros.review_substrate import ReviewRequest
    from ouroboros.usage_accounting import UsageScope, current_usage_scope, usage_scope

    sendable = str(model or row.target_id)
    # The governance tiers take their inline share against the window this row
    # is sent in: the transcript bound derived from the row's window, its
    # output reserve and the owner ceiling (chars), on the
    # `utils.estimate_tokens` scale the tiering budgets with (4 chars a token).
    # A session row's harness model carries no evidenced window, so it resolves
    # the owner ceiling — the one bound that holds for every route.
    ask = ask or SystemReviewAsk()
    task_text, task_facts = _retrieving_task(
        repo_dir, drive_root, required_sources=required_sources, required_sources_ref=required_sources_ref,
        usable_window_tokens=review_native_transcript_bound(
            sendable, output_reserve=_DEEP_MAX_OUTPUT_TOKENS, use_local=row.use_local,
            model_role=f"reviewer:{row.slot_id}",
            credential_profile_id=row.profile_id or None) // 4,
        ask=ask)
    policy = {"output_contract": _REPORT_CONTRACT, "native_data_root": str(drive_root)}
    policy.update(native_required_sources=task_facts["required_sources"],
                  native_required_sources_ref=required_sources_ref or {})
    request = ReviewRequest(
        surface="deep_self_review",
        goal=ask.effective_goal,
        task_id=task_id, call_type="deep_self_review",
        max_tokens=_DEEP_MAX_OUTPUT_TOKENS, no_proxy=True,
        session_root=str(repo_dir), session_task=task_text,
        # The report contract rides the policy with the deep review's header
        # sentence (the shape is `report` either way). The data plane is the
        # REAL runtime root (R5), readable by the reviewer's own tools; memory
        # coverage itself is the inline whitelist in the task (byte-exact,
        # disposition-disclosed) — never receipts.
        policy=policy,
        deadline_at=deadline_at,
    )
    # The logical window: the task's operation window (its finite absolute lifetime,
    # else the operation fallback) narrowed by the owner deadline — the same clock the
    # coordinator gives a slot; without it the native episode would run with no window
    # at all and a session would fall to the transport's own defaults.
    window = review_operation_timeout_sec(
        operation_window_sec(get_task_abs_ceiling_sec()),
        route="agent_session" if row.is_session else "api_chat",
        deadline_at=deadline_at, reserve_sec=get_finalization_grace_sec(),
    )
    from ouroboros.config import review_model_uses_local
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewSlot

    slot = ReviewSlot(
        slot_id=row.slot_id, model=sendable, effort=row_effort(row),
        timeout_sec=window, max_tokens=_DEEP_MAX_OUTPUT_TOKENS,
        role_hint="deep self-reviewer",
        use_local=row.use_local if row.use_local is not None else review_model_uses_local(sendable),
        route=ReviewRouteKind.AGENT_SESSION if row.is_session else ReviewRouteKind.API_CHAT,
        session_target=row.session_target, session_profile=row.profile_id,
        subagent_id=row.subagent_id,
    )
    assignment = ReviewAssignment(
        request=request, slot=slot, call_id=f"deep_self_review:{task_id or 'manual'}",
        call_type="deep_self_review", custody_root=pathlib.Path(drive_root),
    )
    if row.is_session:
        executor = _review_route_executor(assignment, llm=llm)
    else:
        # An api deep_review row IS the bounded inspection episode, whether or
        # not a configured subagent binds it, so a bare route (Main included)
        # never falls back to a one-shot chat with nothing to read.
        from ouroboros.review_native_episode import NativeToolRoundReviewExecutor

        executor = NativeToolRoundReviewExecutor(assignment, llm=llm)
    executor._logical_deadline_monotonic = time.monotonic() + window
    delivery = "agent_session" if row.is_session else "native_tool_rounds"
    emit_progress(
        f"Deep self-review via {delivery} on {sendable}: {_memory_line(task_facts['memory'])}, "
        f"BIBLE.md ({task_facts['bible_chars']:,} chars) delivered inline; window {window:.0f}s..."
    )
    try:
        persist_call(
            pathlib.Path(drive_root), task_id=task_id or "deep_self_review",
            call_id=f"{assignment.call_id}_prompt", call_type="deep_self_review_prompt",
            payload={"request": asdict(request), "slot": asdict(slot), **executor.prompt_payload()},
            manifest={"surface": "deep_self_review", "slot_id": slot.slot_id, "model": slot.model},
        )
    except Exception:
        log.debug("deep self-review prompt custody write failed", exc_info=True)
    scope = _review_usage_scope(current_usage_scope() or UsageScope(), request)
    memory = task_facts["memory"]
    try:
        with usage_scope(scope):
            attempt = executor.execute()
    except Exception as exc:
        # The memory fact precedes EVERY «Выполняется как» record — this
        # failure-custody row included — and rides the typed failure the caller
        # receives (with the executor's proven custody facts, so a failed
        # execution stays visible). BudgetExceeded is recorded, then propagates
        # to the agent's budget rail like every other budget refusal.
        custody = {**executor.failure_custody(), "deep_review_memory": memory}
        _record_execution(slot, custody, status="error", error=f"{type(exc).__name__}: {exc}")
        from ouroboros.llm_claudexor import propagate_model_error
        propagate_model_error(exc)
        if isinstance(exc, BudgetExceeded):
            raise
        log.error("Deep self-review failed: %s", exc, exc_info=True)
        return _failed(f"❌ Deep self-review failed: {type(exc).__name__}: {exc}",
                       reason_code="deep_self_review_error", usage=custody)
    usage = dict(attempt.usage or {})
    # Attached FIRST: the usage handed to every «Выполняется как» record below
    # and the returned usage carry the memory fact. The durable D22 projection
    # itself persists route/model/status/capability_delta and the typed failure
    # facts only — memory is disclosed durably by the header and this usage.
    usage["deep_review_memory"] = memory
    usage["deep_review_governance_manifest"] = task_facts["governance_manifest"]
    # The executor's own list is never mutated (shallow copy): the coverage
    # deltas below are appended to THIS record's copy.
    usage["capability_delta"] = list(usage.get("capability_delta") or [])
    text = str(attempt.raw_text or "")
    if not text.strip():
        # An empty product is an ERROR row in «Выполняется как» — never
        # recorded as a responded review.
        _record_execution(slot, usage, status="error", error="empty response")
        return _failed("⚠️ Model returned an empty response for the deep self-review.",
                       reason_code="deep_self_review_error", usage=usage)
    if delivery == "native_tool_rounds":
        detail = _native_read_coverage(usage, repo_dir)
        usage["deep_review_coverage_basis"] = "source_ranges" if any(c["evidence_basis"] == "source_ranges" for c in detail.values()) else "legacy_lines"
        coverage = {rel: (f"partial({c['fraction']:.2f})" if c["state"] == "partial" else c["state"])
                    for rel, c in detail.items()}
        for rel, c in detail.items():
            if c["state"] not in {"read", "delivered_inline"}:
                covered, total, unit = (c["covered_chars"], c["complete_chars"], "characters") if c["evidence_basis"] == "source_ranges" else (c["covered_lines"], c["total_lines"], "lines")
                missing = f"no delivered range matching the required source revision of {rel}" if c["evidence_basis"] == "source_ranges" else f"no executed repository-root read_file receipt for {rel}"
                unobserved = f"the exact required source extent of {rel} is unobserved" if c["evidence_basis"] == "source_ranges" else f"the {rel} read extent is unobserved (receipts capped or extent not recorded)"
                usage["capability_delta"].append({
                    "kind": "capability_delta",
                    "requested": f"mandatory full read of {rel}",
                    "effective": {
                        "partial": f"{covered} of {total} {unit} of {rel} delivered (merged receipts)",
                        "missing": missing,
                    }.get(c["state"], unobserved),
                    "reason": f"deep_review_mandatory_read_{c['state']}",
                })
    else:
        coverage = {source["path"]: ("delivered_inline" if source.get("coverage_basis") == "delivered_inline"
                                    else "unobserved") for source in task_facts["required_sources"]}
    _record_execution(slot, usage, status="responded")
    try:
        from ouroboros.anthropic_native_custody import public_custody_projection

        persist_call(
            pathlib.Path(drive_root), task_id=task_id or "deep_self_review",
            call_id=f"{assignment.call_id}_response", call_type="deep_self_review_response",
            payload={"message": public_custody_projection(attempt.message), "usage": usage},
            manifest={"surface": "deep_self_review", "slot_id": slot.slot_id, "model": slot.model},
        )
    except Exception:
        log.debug("deep self-review response custody write failed", exc_info=True)
    if not usage.get("capability_delta"):
        usage.pop("capability_delta", None)  # an empty list is not a disclosure
    if not usage.get("resolved_model"):
        usage["resolved_model"] = sendable
    incomplete = _delivery_incomplete(delivery, usage)
    model = str(usage.get("resolved_model") or sendable)
    # Every external value on the HUMAN line is bounded and sanitized too.
    shown_model, shown_reason = _header_value(model), _header_value(incomplete)
    shown_target = _header_value(row.session_target or row.target_id)
    completeness = "complete" if incomplete == "none" else (
        "completeness not host-observed" if incomplete == "unobserved" else f"INCOMPLETE ({shown_reason})")
    if delivery == "native_tool_rounds":
        reads = []
        for rel, c in detail.items():
            covered, total, unit = (c["covered_chars"], c["complete_chars"], "characters") if c["evidence_basis"] == "source_ranges" else (c["covered_lines"], c["total_lines"], "lines")
            reads.append(f"{rel} " + (f"{c['fraction']:.0%} read ({covered}/{total} {unit})" if c["state"] == "partial"
                         else {"read": "read in full", "delivered_inline": "delivered inline in full",
                               "missing": "NOT read"}.get(c["state"], "read extent unobserved")))
        reads = "; ".join(reads)
        human = (
            f"Deep self-review: native inspection episode on {shown_model} — {int(usage.get('native_rounds') or 0)} rounds, "
            f"{int(usage.get('native_tool_calls') or 0)} tool calls ({len(usage.get('native_tool_receipts') or [])} host-observed receipts); "
            f"{reads}; {_memory_line(task_facts['memory'])}; {completeness}"
        )
    else:
        human = (
            f"Deep self-review: agent session {shown_target}"
            + (f" (model {shown_model})" if model and model != (row.session_target or row.target_id) else "")
            + f" — tool reads not host-observed; BIBLE.md delivered inline in full; {_memory_line(task_facts['memory'])}; {completeness}"
        )
    emit_progress(f"Deep self-review complete ({len(text):,} chars; {delivery}, incomplete={incomplete}).")
    return _provenance_header(delivery, model, usage, task_facts["memory"], coverage, human, incomplete=incomplete) + text, usage


def run_deep_self_review(
    repo_dir: pathlib.Path,
    drive_root: pathlib.Path,
    llm: Any,
    emit_progress: Callable[[str], None],
    *,
    task_id: str = "",
    deadline_at: str = "",
    slot: Optional[ConfiguredReviewerSlot] = None,
    required_sources: Optional[list] = None,
    required_sources_ref: Optional[dict] = None,
    ask: Optional[SystemReviewAsk] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Execute the deep self-review on ``slot`` (default: the direct Main row).

    Returns ``(text, usage)``. A delivered report carries the host provenance
    header; every ordinary review failure returns its text with typed usage
    (``execution_status="infra_failed"`` + ``reason_code``) so the caller can
    keep the previous report instead of overwriting it with an error; a failure
    after the task was assembled also carries the memory fact
    (``deep_review_memory``) and the executor's failure custody —
    the same usage its «Выполняется как» error row was recorded from. The ONE
    exception that propagates is ``BudgetExceeded`` — the paid ledger's
    refusal is budget vocabulary for the agent's budget-pause rail, not a
    review error.
    ``slot`` is the row ``/review`` chose (``review_change.system_review_row``);
    ``ask`` is what the call asked beside the standing questionnaire
    (``SystemReviewAsk``: goal, author questions, reason).
    ``required_sources`` and its exact source handle may come from the caller's
    immutable review assembler. Without one, coverage names only the
    constitution actually delivered inline, never an inferred whole-tree scope.
    """
    try:
        row = slot or main_review_row()
        from ouroboros.review_records import apply_review_model_override
        from ouroboros.model_wait import current_model_wait
        waiter = current_model_wait()
        row = apply_review_model_override(row, waiter.overrides) if waiter else row
        reason, model = deep_review_route(row)
        if reason:
            return _failed(deep_review_unavailable_text(reason), reason_code="deep_self_review_unavailable")
        # ONE delivery class: the row retrieves. An api row runs the bounded
        # native inspection episode on the route the availability check
        # resolved (the session keeps its own target as its identity).
        return _run_retrieving_review(
            repo_dir, drive_root, llm, emit_progress, row, task_id=task_id, deadline_at=deadline_at,
            model=row.target_id if row.is_session else str(model or row.target_id),
            required_sources=required_sources, required_sources_ref=required_sources_ref, ask=ask,
        )
    except BudgetExceeded:
        # The paid ledger's refusal is BUDGET vocabulary, not a review error:
        # it propagates so the agent's own `except BudgetExceeded` rail (the
        # budget-pause checkpoint) stays live for the deep review too.
        raise
    except Exception as e:
        from ouroboros.llm_claudexor import propagate_model_error
        propagate_model_error(e)
        log.error("Deep self-review failed: %s", e, exc_info=True)
        return _failed(f"❌ Deep self-review failed: {type(e).__name__}: {e}", reason_code="deep_self_review_error")
