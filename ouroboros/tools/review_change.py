"""``review_change`` — one review-only wave over a change in any registered root.

``commit_reviewed`` is the landing in my own body: it reviews the staged system repository and commits.
``review_change`` is the same review as a capability for any subject I choose — a project copy's
``base..head``, a repository's index or worktree, or the system repository itself. It never commits, never
starts on its own and never runs tests. The rules follow the subject: a subject that is my body is read
against the body layer, anything else against the universal core. Enforcement is recorded for every root
and binds only a body subject's record; a verdict on another root locks nothing.

Panel: the shared strictness policy seats the whole pool for Blocking body changes.
Advisory may narrow with a reason; Cyber Pro composes freely. Enabled unticked rows
are advisers: heard and shown, never counted. Coupling-only advisers answer that
question without acquiring quorum or veto authority.

Surfaces (decision 3A): ``change`` is the panel review above. ``preflight`` is the same wave with ONE
enabled catalog row the author names, pool member or not — the optional pre-commit look at a worktree.
``system`` (``subject=system``) is ``/review``: one row (default: the direct Main row) reads the whole
system and writes a report, recorded as a ``surface=system`` record that asks no verdict question. The
surface is part of every identity, so a preflight record never answers the commit panel.

Two subjects are composed by their own modules and run on this wave: a task result
(``subject=result``, ``tools/review_result.py``) and a skill payload (``subject=skill``,
``tools/review_skill.py``). The skill subject is internal: ``skill_review`` is the model's tool and the
owner's Skills-page action, and it hands this operation the frozen payload it already built
(``_frozen_skill``); the public schema lists no skill subject (owner, 21.10 1A).
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import json
import logging
import pathlib
import subprocess
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from ouroboros.config import (
    adaptive_quorum, get_finalization_grace_sec, get_llm_transport_read_timeout_sec, get_review_enforcement,
    get_runtime_mode, get_task_abs_ceiling_sec, operation_window_sec,
)
from ouroboros.preflight_receipts import bind_execution_exhibit
from ouroboros.review_body_fact import body_fact, layer_for
from ouroboros.review_ledger import (
    PART_CHANGE, PART_RESULT, PART_SKILL, PARTS, QUESTION_NOT_PERFORMED, build_wave_record, ledger_root, load_record, new_record_id,
    panel_facts, panel_not_reached, reduce_verdict, revise_record, row_verdict, seat_findings, write_record,
)
from ouroboros.settings_scales import EFFORT_SCALE
from ouroboros.reviewer_slot_config import ComposedPanel, PanelCompositionError as ReviewChangeArgumentError
from ouroboros.tools.arg_feedback import argument_refusal
from ouroboros.tools.parallel_review import run_parallel_review
from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.tools.review_change_custody import (
    arm_rejoin, install_paid_stamp, pending_round_attempt, settle_attempt,
)
from ouroboros.tools.review_helpers import checklist_fingerprint
from ouroboros.tools.review_subject import (
    ReviewSubjectSpec, checkout_token, freeze_subject, is_gate_subject, isolated_checkout, reuse_or_none,
    review_retry_key, review_reuse_key, review_round_sha,
)
from ouroboros.utils import run_cmd, utc_now_iso

log = logging.getLogger(__name__)

TOOL_NAME = "review_change"
# A result wave keeps its own paid-cycle count per task tree (attempt rows carry this tool
# name), as task acceptance did: change reviews never use up an owed result review. A skill
# wave likewise keeps its own (``review_skill``: per root task, or per content version in the
# manual lane, owner Q23).
RESULT_TOOL_NAME = "review_result"
SKILL_TOOL_NAME = "review_skill"
SURFACE = "change"
RESULT = "result"
SKILL = "skill"
SINGLE_QUESTION = (RESULT, SKILL)
SURFACES = ("change", "preflight", "system", RESULT, SKILL)
ROOTS = ("active_workspace", "system_repo")
CHANGE_KINDS = ("index", "worktree", "base..head")
SUBJECT_KINDS = (*CHANGE_KINDS, "system", RESULT, SKILL)
# What the model's tool offers: the skill subject is ``skill_review``'s own spelling.
PUBLIC_SURFACES = tuple(surface for surface in SURFACES if surface != SKILL)
PUBLIC_SUBJECT_KINDS = tuple(kind for kind in SUBJECT_KINDS if kind != SKILL)
# The part a system review's one seat answers: a report, outside the verdict
# questions (``review_ledger.PARTS``), so its record never states a verdict.
REPORT_PART = "report"
MAIN_SEAT = "main"
# Wave block reasons the seat rows state themselves (findings, quorum, coupling);
# any other block keeps a verdict recomputed over the assigned seats from PASS.
_ROW_STATED_BLOCKS = frozenset({"", "critical_findings", "review_quorum", "coupling_not_performed", "change_unanswered"})
_GIT_PROBE_TIMEOUT_SEC = 30


# --- Arguments ------------------------------------------------------------------

@dataclass(frozen=True)
class ReviewChangeRequest:
    root: str = "active_workspace"
    workspace_root: str = ""
    subject: str = ""
    base: str = ""
    head: str = ""
    goal: str = ""
    scope: str = ""
    author_questions: Tuple[str, ...] = ()
    reviewers: Tuple[str, ...] = ()
    reason: str = ""
    coupling_only: Tuple[str, ...] = ()
    reviewer_effort: str = ""
    review_rebuttal: str = ""
    treat_as_body: bool = False
    surface: str = SURFACE
    answer: str = ""  # subject=result: the reply under review (omitted = the task's latest reply)
    since_review: str = ""  # a change review record to compare against (review_history.subject_comparison)


_ARG_NAMES = frozenset(ReviewChangeRequest.__dataclass_fields__)
_TEXT_ARGS = ("root", "workspace_root", "subject", "base", "head", "surface", "goal", "scope",
              "reason", "reviewer_effort", "review_rebuttal", "answer", "since_review")


def wave_tool_name(request: Any) -> str:
    """The attempt rows' tool name: one paid-cycle count per kind of review."""
    surface = str(getattr(request, "surface", "") or "")
    return RESULT_TOOL_NAME if surface == RESULT else SKILL_TOOL_NAME if surface == SKILL else TOOL_NAME


def _text(args: Dict[str, Any], name: str, problems: List[str]) -> str:
    value = args.get(name)
    if value is not None and not isinstance(value, str):
        problems.append(f"{name} must be a string")
    return value if isinstance(value, str) else ""


def _strings(args: Dict[str, Any], name: str, problems: List[str], *, verbatim: bool) -> Tuple[str, ...]:
    value = args.get(name) or []
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
        problems.append(f"{name} must be a list of strings")
        return ()
    kept = [item if verbatim else item.strip() for item in value if item.strip()]
    return tuple(kept) if verbatim else tuple(dict.fromkeys(kept))


def parse_request(args: Dict[str, Any]) -> ReviewChangeRequest:
    """The call's arguments as one request; every violated rule is named at once."""
    problems: List[str] = []
    unknown = sorted(set(args) - _ARG_NAMES)
    if unknown:
        problems.append(f"unknown argument(s) {', '.join(unknown)}")
    text = {name: _text(args, name, problems) for name in _TEXT_ARGS}
    root = text["root"].strip() or "active_workspace"
    subject, base, head = text["subject"].strip(), text["base"].strip(), text["head"].strip()
    surface = text["surface"].strip() or SURFACE
    if subject in SINGLE_QUESTION and surface == SURFACE:
        surface = subject  # the subject names its surface, as /review's does
    lists = {name: _strings(args, name, problems, verbatim=name == "author_questions")
             for name in ("author_questions", "reviewers", "coupling_only")}
    if root not in ROOTS:
        problems.append(f"root must be one of {', '.join(ROOTS)} (a registered folder goes in workspace_root), got {root!r}")
    if root == "system_repo" and text["workspace_root"].strip():
        problems.append("root=system_repo names the system repository; leave workspace_root empty")
    if surface not in SURFACES:
        problems.append(f"surface must be one of {', '.join(SURFACES)}, got {surface!r}")
    if subject not in SUBJECT_KINDS:
        problems.append(f"subject must be one of {', '.join(SUBJECT_KINDS)}, got {subject!r}")
    elif (subject == "system") != (surface == "system"):
        problems.append("subject=system goes with surface=system: /review reads the whole system repository")
    elif (subject == RESULT) != (surface == RESULT):
        problems.append("subject=result goes with surface=result: it reviews this task's result, not a change")
    elif (subject == SKILL) != (surface == SKILL):
        problems.append("subject=skill goes with surface=skill: a skill payload is reviewed through skill_review")
    elif subject in SINGLE_QUESTION and (text["workspace_root"].strip() or base or head or root != "active_workspace"):
        problems.append(f"subject={subject} reviews its own frozen subject; root, workspace_root, base and head do not apply")
    elif subject == "system" and (text["workspace_root"].strip() or base or head):
        problems.append("subject=system reads the installed system repository; workspace_root, base and head do not apply")
    elif subject == "base..head" and not (base and head):
        problems.append("subject=base..head needs both base and head revisions")
    elif subject in ("index", "worktree") and head:
        problems.append(f"head belongs to subject=base..head; subject={subject} is reviewed against base (default HEAD)")
    for name, rev in (("base", base), ("head", head)):
        if rev.startswith("-"):
            problems.append(f"{name} must name a revision, not an option ({rev!r})")
    if surface in ("preflight", "system") and (lists["coupling_only"] or len(lists["reviewers"]) > 1
                                               or (surface == "preflight" and not lists["reviewers"])):
        problems.append(f"surface={surface} seats exactly one reviewer, named in reviewers"
                        + (" (omitted = the direct Main row)" if surface == "system" else "")
                        + "; coupling_only does not apply")
    if subject in SINGLE_QUESTION and lists["coupling_only"]:
        problems.append(f"coupling_only belongs to a change review; a {subject} review asks one question of every seat")
    if text["answer"].strip() and subject != RESULT:
        problems.append("answer belongs to subject=result")
    if text["since_review"].strip() and subject not in CHANGE_KINDS:
        problems.append("since_review belongs to a change review (subject index, worktree or base..head)")
    effort = text["reviewer_effort"].strip()
    if effort and effort not in EFFORT_SCALE:
        problems.append(f"reviewer_effort must be one of {', '.join(EFFORT_SCALE)}, got {effort!r}")
    treat = args.get("treat_as_body", False)
    if not isinstance(treat, bool):
        problems.append("treat_as_body must be a boolean")
    if problems:
        raise ReviewChangeArgumentError("; ".join(problems))
    return ReviewChangeRequest(
        root="system_repo" if subject == "system" else root, workspace_root=text["workspace_root"].strip(),
        subject=subject, base=base, head=head,
        goal=text["goal"], scope=text["scope"], author_questions=lists["author_questions"],
        reviewers=lists["reviewers"], reason=text["reason"], coupling_only=lists["coupling_only"],
        reviewer_effort=effort, review_rebuttal=text["review_rebuttal"],
        treat_as_body=treat is True and subject not in SINGLE_QUESTION,
        answer=text["answer"], since_review=text["since_review"].strip(),
        surface=surface)


# --- Root -------------------------------------------------------------------------

def _git_line(cwd: pathlib.Path, *args: str) -> str:
    try:
        return run_cmd(["git", *args], cwd=cwd, timeout=_GIT_PROBE_TIMEOUT_SEC).strip()
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError):
        return ""


def _system_repo(ctx: ToolContext) -> pathlib.Path:
    """The context's body root: the system repository, or the bound body candidate
    (``body_candidate.bind``) that this task authors — what ``root=system_repo`` names."""
    from ouroboros.tools.tool_resolution import system_repo_dir_for

    return pathlib.Path(system_repo_dir_for(ctx)).resolve()


def _governance_repo(ctx: ToolContext) -> pathlib.Path:
    """The INSTALLED body whose rules execute — governance for every subject and the
    body identity the predicate compares against. For a bound body candidate this is
    the SERVING checkout, never the candidate (``review_substrate.review_repo_dirs_for``'s
    rule: the candidate is the subject; the body that is running is the authority)."""
    from ouroboros.body_candidate import serving_repo_dir_for

    return pathlib.Path(serving_repo_dir_for(ctx)).resolve()


def _admitted_workspace_root(ctx: ToolContext, value: str) -> pathlib.Path:
    """``schedule_subagent``'s ``workspace_root`` policy: an existing folder the
    caller can already read, relative paths against its available workspace."""
    from ouroboros.tool_access_reads import admit_child_start_folder, capture_parent_workspace

    try:
        path = pathlib.Path(value).expanduser()
        if not path.is_absolute():
            parent = capture_parent_workspace(ctx)
            if not parent.get("root") or parent.get("availability") == "unavailable":
                raise ValueError("a relative workspace_root needs an available workspace; name an absolute folder")
            path = pathlib.Path(parent["root"]) / path
        return pathlib.Path(admit_child_start_folder(ctx, path, {}))
    except (OSError, ValueError, RuntimeError) as exc:
        raise ReviewChangeArgumentError(f"workspace_root {value!r} is not a registered folder this task can read: {exc}") from exc


def resolve_review_root(ctx: ToolContext, binding: Any, request: ReviewChangeRequest) -> Tuple[str, pathlib.Path]:
    """(root_kind, top level) of the repository this call reviews."""
    from ouroboros.tools import git as git_mod

    if request.workspace_root:
        start = _admitted_workspace_root(ctx, request.workspace_root)
    else:
        try:
            start = pathlib.Path(git_mod._vcs_binding(ctx, binding, root=request.root).base_path)
        except (OSError, ValueError, RuntimeError) as exc:
            raise ReviewChangeArgumentError(f"root={request.root} does not resolve here: {exc}") from exc
    top = _git_line(start, "rev-parse", "--show-toplevel")
    if not top:
        raise ReviewChangeArgumentError(
            f"{start} is not inside a git repository; name a registered repository with workspace_root, or root=system_repo")
    root = pathlib.Path(top).resolve()
    assigned = _assigned_review_root(ctx)
    if assigned is not None and not root.is_relative_to(assigned):
        raise ReviewChangeArgumentError(f"review_change is limited to the assigned work at {assigned}; got {root}")
    return ("system_repo" if root == _system_repo(ctx) else "active_workspace"), root


def _assigned_review_root(ctx: ToolContext) -> Optional[pathlib.Path]:
    """Ordinary working children may ask reviewers about their own physical work.

    Read access to another registered folder grants no review authority there.
    Cyber keeps its existing authority; a read-only child stays read-only.
    """
    from ouroboros.contracts.task_constraint import normalize_task_constraint
    from ouroboros.runtime_mode_policy import mode_has_unrestricted_agency
    from ouroboros.tool_access import active_tool_profile

    profile = active_tool_profile(ctx)
    if profile == "local_readonly_subagent":
        raise ReviewChangeArgumentError("a read-only child cannot dispatch review_change")
    if profile != "acting_subagent" or mode_has_unrestricted_agency(get_runtime_mode()):
        return None
    constraint = normalize_task_constraint(getattr(ctx, "task_constraint", None))
    if not constraint or not constraint.write_root:
        raise ReviewChangeArgumentError("review_change requires an assigned write root")
    return pathlib.Path(constraint.write_root).resolve()


def _verify_revisions(root: pathlib.Path, request: ReviewChangeRequest) -> None:
    for name in ("base", "head"):
        rev = getattr(request, name)
        if rev and not _git_line(root, "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}"):
            raise ReviewChangeArgumentError(f"{name} {rev!r} is not a commit in {root}")


# --- Panel ------------------------------------------------------------------------

def compose_panel(request: ReviewChangeRequest, *, body: bool = False, strictness: Optional[str] = None,
                  required_root: bool = False, chosen_by: str = "ouroboros") -> ComposedPanel:
    """The shared strictness policy, applied to this change's own-body fact.

    ``required_root`` is the host fact that this result wave is a Required root
    (``review_result.required_root_wave``): the surface becomes ``result_required``
    (Blocking widens, Advisory refuses an unexplained narrowing). Auto, a child and
    Cyber Pro leave it ``result`` and compose the named rows. It is never a model
    argument. ``chosen_by`` is the caller's record of who seated the panel (the host
    floor passes ``host``). A skill wave composes as the skill surface. Every seat of
    a result or skill wave answers its one question.
    """
    from ouroboros.reviewer_slot_config import compose_review_panel, review_strictness

    surface = ("preflight" if request.surface == "preflight"
               else "result_required" if request.surface == RESULT and required_root
               else request.surface if request.surface in SINGLE_QUESTION
               else "own_body" if body else "change")
    try:
        panel = compose_review_panel(request.reviewers, request.reason, surface=surface,
            strictness=strictness or review_strictness(get_review_enforcement(), get_runtime_mode()),
            reviewer_effort=request.reviewer_effort, coupling_only=request.coupling_only,
            chosen_by=chosen_by or "ouroboros")
    except ReviewChangeArgumentError:
        raise
    except ValueError as exc:
        return ComposedPanel((), True, {"composition": "full_pool", "composition_error": str(exc)})
    return single_question_panel(panel, request.surface)


def single_question_panel(panel: ComposedPanel, surface: str) -> ComposedPanel:
    """A single-question wave's seats answer its one part (``result`` / ``skill``), whoever composed them."""
    from ouroboros.reviewer_slot_config import PoolSeat

    if surface not in SINGLE_QUESTION:
        return panel
    part = PART_RESULT if surface == RESULT else PART_SKILL
    return ComposedPanel(tuple(PoolSeat(seat.slot, (part,), seat.additional) for seat in panel.seats), True, panel.facts)


@contextlib.contextmanager
def _panel_in_force(panel: ComposedPanel) -> Iterator[None]:
    if not panel.override:
        yield
        return
    from ouroboros.reviewer_slot_config import composed_review_pool

    with composed_review_pool(panel.seats):
        yield


# --- The wave ---------------------------------------------------------------------

@dataclass(frozen=True)
class _Wave:
    request: ReviewChangeRequest
    frozen: Any
    panel: ComposedPanel
    root: pathlib.Path
    fact: Any
    layer: str
    rules: Dict[str, Any]
    enforcement: str
    contract_fp: str
    rebuttal_sha: str
    retry_key: str
    reuse_key: str
    root_task_id: str
    record_id: str
    label: str
    # The open attempt row of this same round on this task that the wave collects
    # instead of paying again (review_change_custody.pending_round_attempt); None = new.
    rejoin: Any = None
    # The test execution exhibit an in-process caller supplied, bound to ``frozen``
    # (preflight_receipts.bind_execution_exhibit); a tool call never supplies one.
    execution: Dict[str, Any] = dataclasses.field(default_factory=lambda: {"status": "unobserved"})


def _dispatched_execution(ctx: ToolContext, record_id: str, attempt: Any) -> Dict[str, Any]:
    """The original exhibit, from the ledger or the paid request if dispatch crashed
    before the ledger write. Legacy absence stays unobserved, never today's fact."""
    from ouroboros.review_evidence import pending_commit_review_evidence

    try:
        record = load_record(ledger_root(ctx), record_id) or {}
    except (OSError, ValueError):
        record = {}
    if record:
        return dict(record.get("preflight_execution") or {"status": "unobserved"})
    collector = SimpleNamespace(**vars(ctx))
    collector._pending_review_attempt = attempt
    collector._current_review_retry_key = attempt.review_retry_key
    return pending_commit_review_evidence(collector, "preflight_execution") or {"status": "unobserved"}


def _round_sha(request: ReviewChangeRequest, frozen: Any) -> str:
    """Identity (b) of this request over this frozen subject."""
    from ouroboros.tools.commit_gate import compute_rebuttal_sha256

    return review_round_sha(frozen, rebuttal_sha=str(compute_rebuttal_sha256(request.review_rebuttal) or ""),
                            questions=request.author_questions, goal=request.goal, scope=request.scope,
                            since_review=request.since_review)


def wave_label_material() -> list:
    """Review-contract material (``commit_gate.commit_review_contract_fingerprint``): the label a change or
    preflight wave hands its seats as its commit message, for every subject kind, against a base or not."""
    from types import SimpleNamespace

    return [_wave_label(SimpleNamespace(surface=surface, subject=subject, base="⟨base⟩" if against else "",
                                        head="⟨head⟩"), pathlib.Path("⟨root⟩"))
            for surface in (SURFACE, "preflight") for subject in ("base..head", "index", "worktree")
            for against in (False, True)]


def _wave_label(request: ReviewChangeRequest, root: pathlib.Path) -> str:
    if request.surface == RESULT:
        return "review_change result: this task's result against what was asked (review only; nothing is delivered)"
    if request.surface == SKILL:
        return f"skill_review: the payload of skill {root.name} (review only; nothing is enabled or granted)"
    span = {"base..head": f"{request.base}..{request.head}", "index": "staged index", "worktree": "worktree"}
    against = f" against {request.base}" if request.base and request.subject != "base..head" else ""
    surface = "" if request.surface == SURFACE else f" {request.surface}"
    return f"review_change{surface}: {span[request.subject]}{against} of {root.name} (review only; nothing is committed)"


def _prepare_wave(ctx: ToolContext, request: ReviewChangeRequest, frozen: Any, panel: ComposedPanel, *,
                  root: pathlib.Path, fact: Any, layer: str, rejoin: Any = None) -> _Wave:
    """Every identity of this wave, computed under its own panel."""
    from ouroboros.tools.commit_gate import commit_review_contract_fingerprint, compute_rebuttal_sha256, resolve_root_task_id

    if request.surface == RESULT:
        from ouroboros.tools.review_result import result_contract_fingerprint, result_rules

        rules, contract = result_rules(), result_contract_fingerprint
    elif request.surface == SKILL:
        from ouroboros.tools.review_skill import skill_contract_fingerprint, skill_rules

        rules, contract = skill_rules(), skill_contract_fingerprint
    else:
        # A change or preflight wave is held to its layer's change brief, as the commit gate is to the body's.
        rules, contract = checklist_fingerprint(layer), functools.partial(commit_review_contract_fingerprint, layer=layer)
    enforcement = str(get_review_enforcement() or "")
    # A rejoin collects under the contract its dispatch recorded. The rules stay the record's
    # provenance (``_finish_record``), never identity (a): governance text buys no new wave.
    contract_fp = str(rejoin.review_contract_fingerprint or "") if rejoin is not None else str(contract() or "")
    rebuttal_sha = str(compute_rebuttal_sha256(request.review_rebuttal) or "")
    # Identity (b), the logical round, enters both the reuse key (a) and the custody
    # retry key (c): a new rebuttal/question/brief/revision pair is a new wave and a
    # new physical operation; a retry of the same round rejoins the old one.
    round_sha = _round_sha(request, frozen)
    reuse_key = review_reuse_key(
        frozen, layer=layer, assigned=panel.assigned, enforcement=enforcement, contract_fp=contract_fp,
        round_sha=round_sha)
    # A subject may name the root task its paid-cycle allowance belongs to ("" = the manual lane,
    # counted per content version): a skill reviewed from the Skills page is nobody's task.
    ceiling_root = getattr(frozen, "ceiling_root", None)
    return _Wave(request=request, frozen=frozen, panel=panel, root=root, fact=fact, layer=layer, rules=rules,
                 enforcement=enforcement, contract_fp=contract_fp, rebuttal_sha=rebuttal_sha,
                 retry_key=review_retry_key(frozen, round_sha=round_sha), reuse_key=reuse_key,
                 root_task_id=str(ceiling_root if ceiling_root is not None else resolve_root_task_id(ctx) or ""),
                 record_id=new_record_id(), label=_wave_label(request, root))


# The shared task context's review state; the wave runs with this call's
# identities and a fresh history, and every field is restored afterwards.
_CTX_FIELDS = (
    "last_push_succeeded", "_review_advisory", "_last_triad_models", "_last_coupling_result",
    "_last_triad_raw_results", "_last_review_critical_findings", "_last_review_block_reason",
    "_last_review_advisory_findings", "_last_review_verdict", "_last_scope_raw_result",
    "_last_review_structured", "_review_degraded_reasons", "_current_review_tool_name",
    "_current_review_retry_key", "_current_review_record_id", "_review_reconcile_only",
    "_review_frozen_rows", "_review_custody_lost", "_current_review_attempt_number",
    "_author_commit_source", "_author_commit_decision", "_author_commit_record", "_commit_review_status",
    "_current_review_rebuttal_sha256", "_current_review_contract_fingerprint", "_review_history",
    "_review_iteration_count", "_coupling_review_history", "_coupling_review_history_rounds",
    "_triad_withheld_seat_records",
    "_review_paid_stamp", "_review_reserved_roster", "_review_reserved_operations",
    "_review_pending_invocation_checkpoint", "_last_review_slot_executions", "_pending_review_attempt",
    "_commit_preflight", "_commit_review_panel",
    "_review_dispute_history", "_review_dispute_input", "_preflight_execution", "_review_since",
)


@contextlib.contextmanager
def _wave_context(ctx: ToolContext, wave: _Wave) -> Iterator[None]:
    from ouroboros.tools import git as git_mod

    missing = object()
    saved = {name: getattr(ctx, name, missing) for name in _CTX_FIELDS}
    try:
        git_mod._reset_commit_review_state(ctx)
        ctx._current_review_tool_name = wave_tool_name(wave.request)
        ctx._current_review_retry_key = wave.retry_key
        ctx._current_review_record_id = wave.record_id
        ctx._current_review_rebuttal_sha256 = wave.rebuttal_sha
        ctx._current_review_contract_fingerprint = wave.contract_fp
        ctx._review_history, ctx._review_iteration_count, ctx._coupling_review_history = [], 0, {}
        ctx._review_dispute_history, ctx._review_dispute_input = None, None
        ctx._preflight_execution, ctx._review_since = wave.execution, wave.request.since_review
        from ouroboros import review_pause
        if wave.rejoin is not None:
            with review_pause.receptive_author(review_pause.rejoin_binding(ctx, wave.record_id)):
                yield
        else:
            yield
    finally:
        for name, value in saved.items():
            if value is missing:
                with contextlib.suppress(AttributeError):
                    delattr(ctx, name)
            else:
                setattr(ctx, name, value)


def _dispatch(ctx: ToolContext, wave: _Wave) -> Dict[str, Any]:
    """Run the one wave; the paid stamp records it at its first physical dispatch.
    A rejoin runs the same wave on the gate's reconcile-only path (nothing new is
    sent; the open operation is collected under the attempt row it opened)."""
    from ouroboros.tools import git as git_mod
    from ouroboros.tools.review_helpers import goal_with_author_questions

    holder = install_paid_stamp(ctx, wave)
    arm_rejoin(ctx, wave)
    goal = goal_with_author_questions(wave.request.goal, wave.request.author_questions)
    try:
        review_err, _coupling, block_reason, _advisory = run_parallel_review(
            ctx, wave.label, goal=goal, scope=wave.request.scope,
            review_rebuttal=wave.request.review_rebuttal,
            review_binding_fingerprint=str(wave.frozen.diff_sha), subject=wave.frozen)
    except Exception as exc:  # the wave's own failure is a typed infra block, never a verdict
        log.warning("review_change wave failed", exc_info=True)
        return {"blocked": True, "block_reason": "infra_failure", "crash": f"{type(exc).__name__}: {exc}",
                "attempt": holder}
    finally:
        git_mod._reconcile_and_clear_review_roster(ctx)
    # The one wave decides once (review_ledger.reduce_verdict): a coupling FAIL is
    # a critical finding of the same verdict, never a second block.
    if review_err:
        return {"blocked": True, "block_reason": str(block_reason or ""), "attempt": holder}
    return {"blocked": False, "block_reason": "", "attempt": holder}


def _forensic(ctx: ToolContext) -> Dict[str, Any]:
    return {
        "dispute_input": getattr(ctx, "_review_dispute_input", None),
        "structured": dict(getattr(ctx, "_last_review_structured", {}) or {}),
        "triad_raw": [row for row in (getattr(ctx, "_last_triad_raw_results", []) or []) if isinstance(row, dict)],
        "degraded_reasons": [str(item) for item in (getattr(ctx, "_review_degraded_reasons", []) or [])],
        "block_reason": str(getattr(ctx, "_last_review_block_reason", "") or ""),
        "custody_lost": bool(getattr(ctx, "_review_custody_lost", False)),
        # This wave's seat executions (``reviewer_slot_config.record_reviewer_slot_execution``
        # keeps them on the wave's ctx), never the process-wide last-execution projection.
        "slot_executions": dict(getattr(ctx, "_last_review_slot_executions", {}) or {}),
        "preflight_execution": dict(getattr(ctx, "_preflight_execution", None) or {"status": "unobserved"}),
    }


def _open_seats(forensic: Dict[str, Any]) -> List[str]:
    """Seats whose answer may still arrive. A seat whose custody is lost is terminal ("outcome
    unknown, never re-sent"): it is a failed seat of a settled record, never a reason to keep
    the record open for a re-issue that cannot change it."""
    rows = list(forensic.get("triad_raw", []))
    return [str(row.get("slot_id") or "") for row in rows
            if str(row.get("operation_state") or "") != "custody_lost"
            and (bool(row.get("late_result_pending")) or str(row.get("operation_state") or "") == "in_flight")]


def _seats_open(forensic: Dict[str, Any]) -> bool:
    return bool(_open_seats(forensic))


def _checkout_custody(ctx: ToolContext, forensic: Dict[str, Any]) -> Dict[str, Any]:
    """What keeps the isolated checkout alive after the wave, ``{}`` when nothing:
    a seat whose answer is still owed (late, in flight, custody lost) or an open
    preflight run on this context — read INSIDE the wave context, where the wave's
    own review state is still on ``ctx`` (``git_review_cycle._review_custody_pending``)."""
    from ouroboros.tools import git as git_mod

    facts: Dict[str, Any] = {}
    seats = _open_seats(forensic)
    if seats:
        facts["seats"] = seats
    if forensic.get("custody_lost"):
        facts["custody_lost"] = True
    if git_mod._review_custody_pending(ctx):
        facts["review_custody_pending"] = True
    return facts


def wave_facts(ctx: ToolContext, wave: _Wave, *, outcome: Dict[str, Any], forensic: Dict[str, Any]) -> Dict[str, Any]:
    """The ledger facts of this wave, in the commit gate's vocabulary."""
    from ouroboros.review_records import ReviewRequest, resolve_review_wave
    from ouroboros.tools import git as git_mod
    from ouroboros.tools.review_helpers import review_enforcement_blocks

    task_id = str(getattr(ctx, "task_id", "") or "")
    structured = dict(forensic.get("structured") or {})
    triad_raw = list(forensic.get("triad_raw") or [])
    refusal = outcome.get("dispatch_refusal")
    if refusal is None and forensic.get("block_reason") == "review_wave_budget_insufficient":
        refusal = {"kind": "review_wave_budget_insufficient", "message": str(structured.get("wave_refusal") or "")}
    degraded = list(forensic.get("degraded_reasons") or [])
    if outcome.get("crash"):
        degraded.append(f"review_change_wave_crashed: {outcome['crash']}")
    critical, advisory, additional = seat_findings(triad_raw, set(wave.panel.additional))
    executions = dict(forensic.get("slot_executions") or {})
    mode = ""
    with contextlib.suppress(Exception):
        mode = str(git_mod._current_runtime_mode() or "")
    return {
        "task_id": task_id, "root_task_id": wave.root_task_id,
        "review_wave_id": resolve_review_wave(ReviewRequest(
            surface=wave.request.surface, goal=wave.request.goal or wave.label, task_id=task_id,
            retry_key=wave.retry_key), {}, ""),
        "subject": wave.frozen, "repo_dir": str(wave.root), "commit_message": wave.label,
        "governance_root": str(wave.frozen.spec.governance_root or ""), "layer": wave.layer,
        "body_fact": str(wave.fact.body), "body_how": str(wave.fact.how),
        "goal": wave.request.goal, "scope": wave.request.scope, "author_questions": list(wave.request.author_questions),
        "since_review": wave.request.since_review,
        "binding_fingerprint": str(wave.frozen.diff_sha), "review_contract_fingerprint": wave.contract_fp,
        "rebuttal_sha256": wave.rebuttal_sha, "enforcement": wave.enforcement, "mode": mode,
        "dispute_input": forensic.get("dispute_input"),
        "enforcement_blocks": wave.layer == "body" and bool(review_enforcement_blocks(wave.enforcement)),
        "structured": structured, "slot_executions": executions, "triad_raw": triad_raw,
        # The wave's "pending" block counts a lost seat as still owed; once no seat can still answer
        # (``_open_seats``), the record is decided by its answers, never held by that block.
        "blocked": bool(outcome.get("blocked")) and not (
            str(outcome.get("block_reason") or "") == "review_late_result_pending" and not _seats_open(forensic)),
        "block_reason": str(outcome.get("block_reason") or ""),
        "dispatch_refusal": refusal, "pending": _seats_open(forensic), "lost_seat_terminal": True,
        "degraded_reasons": degraded,
        "critical_findings": critical, "advisory_findings": advisory, "additional_findings": additional,
        "tests": {"policy": "NOT_RUN", "result": "unknown"},
        "preflight": {"status": "not_performed", "record_id": ""},
        "preflight_execution": dict(forensic.get("preflight_execution") or {"status": "unobserved"}),
        "reuse_key": wave.reuse_key if refusal is None else "", "retry_key": wave.retry_key,
        "composition": str(wave.panel.facts.get("composition") or "full_pool"),
        "composition_reason": str(wave.panel.facts.get("reason") or ""),
        "chosen_by": str(wave.panel.facts.get("chosen_by") or "owner"),
    }


def subject_facts(frozen: Any, retention: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The record's subject; ``retained_checkout`` names the isolated checkout kept
    alive by open custody (``retention`` says what), ``""`` when it was removed."""
    spec = frozen.spec
    checkout = str(getattr(frozen, "checkout", "") or "")
    extra = getattr(frozen, "extra_subject_facts", None)  # a subject may state more identity than the git keys
    return {"root_kind": str(spec.root_kind), "root": str(spec.root), "kind": str(spec.kind),
            "base": str(spec.base or getattr(frozen, "parent_sha", "") or ""), "head": str(spec.head or ""),
            "governance_root": str(getattr(spec, "governance_root", "") or ""),
            "tree_sha": str(getattr(frozen, "tree_sha", "") or ""), "diff_sha": str(getattr(frozen, "diff_sha", "") or ""),
            "checkout": checkout, "retained_checkout": checkout if retention else "",
            "retention": dict(retention or {}), **(dict(extra() or {}) if callable(extra) else {})}


def _assigned_verdict(record: Any, rows: List[Dict[str, Any]], outcome: Dict[str, Any]) -> Dict[str, Any]:
    """The decision over the ASSIGNED seats only; added seats never count. The
    quorum is ``adaptive_quorum`` of the assigned seats — one number, as the gate's."""
    from ouroboros.review_ledger import seat_open

    reason = str(outcome.get("block_reason") or "")
    verdict = reduce_verdict(
        rows, quorum_required=adaptive_quorum(len(rows)) if rows else 0,
        gate_blocked=bool(outcome.get("blocked")) and reason not in _ROW_STATED_BLOCKS, gate_reason=reason,
        dispatch_refusal=record.dispatch_refusal, pending=any(seat_open(row) for row in rows))
    verdict["per_row"] = {**{str(s.get("seat_id") or ""): row_verdict(s) for s in record.rows}, **verdict["per_row"]}
    return verdict


def _finish_record(record: Any, wave: _Wave, outcome: Dict[str, Any], *, reuse_key: str,
                   retention: Optional[Dict[str, Any]] = None) -> None:
    record.subject = {**dict(record.subject or {}), **subject_facts(wave.frozen, retention)}
    record.brief = {**dict(record.brief or {}), "author_questions": list(wave.request.author_questions),
                    "checklist": {"layer": wave.layer, "body_fact": str(wave.fact.body), "how": str(wave.fact.how),
                                  "treat_as_body": wave.request.treat_as_body,
                                  "checklist_hash": str(wave.rules.get("checklist_hash") or ""),
                                  "rules_source": dict(wave.rules.get("rules_source") or {})}}
    brief_facts = getattr(wave.frozen, "brief_facts", None)  # a subject's own brief facts (a skill's lifecycle, rule)
    for key, value in (dict(brief_facts() or {}) if callable(brief_facts) else {}).items():
        if key == "checklist":
            record.brief["checklist"].update(dict(value or {}))
        else:
            record.brief[key] = value
    record.fingerprints = {**dict(record.fingerprints or {}), "reuse_key": reuse_key, "retry_key": wave.retry_key}
    additional = set(wave.panel.additional)
    for seat in record.rows:
        seat["additional"] = str(seat.get("seat_id") or "") in additional
    panel = dict(record.panel or {})
    if additional and record.rows:
        # The verdict and its quorum are the ASSIGNED seats' alone; the panel block
        # describes everyone who sat (``panel_facts`` counts the added seats apart).
        verdict = _assigned_verdict(record, [seat for seat in record.rows if not seat["additional"]], outcome)
        record.verdict = {**dict(record.verdict or {}), **verdict}
        record.brief["parts"] = [part for part in PARTS if verdict["per_question"].get(part) != QUESTION_NOT_PERFORMED]
        panel = panel_facts(record.rows, composition=str(wave.panel.facts.get("composition") or "full_pool"),
                            reason=str(wave.panel.facts.get("reason") or ""),
                            chosen_by=str(wave.panel.facts.get("chosen_by") or "owner"))
    record.panel = {**panel, **wave.panel.facts, "not_reached": panel_not_reached(record.rows, lost_is_terminal=True)}
    if wave.request.surface == "preflight":
        for key in ("pool", "strictness", "pool_empty", "main_self_review", "reason_missing"):
            record.panel.pop(key, None)


def _write(drive: Any, record: Any, *, rejoin: bool = False) -> Tuple[Dict[str, Any], bool]:
    """Write the one record (a rejoin revises the record its open operation left pending, keeping what that
    dispatch's seats read: the subject's identity, the brief and the fingerprints with the reuse key over its
    rebuttal; the seats' answers, refs and custody facts are the settling wave's); an unwritten record is
    answered from memory and says so."""
    def settle(prior: Dict[str, Any]) -> Dict[str, Any]:
        from ouroboros.review_ledger import STATE_SETTLED, settle_pending_payload

        fresh = record.to_dict()
        merged = settle_pending_payload(prior, fresh)
        if fresh.get("state") == merged.get("state") == STATE_SETTLED:
            # Checkout retention is custody, not the immutable reviewed subject.
            merged["subject"].update({key: fresh["subject"][key] for key in
                                      ("retained_checkout", "retention") if key in fresh.get("subject", {})})
        return merged

    try:
        revised = revise_record(drive, record.record_id, settle) if rejoin else None
        return revised or write_record(drive, record), True
    except (OSError, ValueError) as exc:
        log.warning("review_change ledger record was not written", exc_info=True)
        payload = record.to_dict() if hasattr(record, "to_dict") else dict(record)
        verdict = dict(payload.get("verdict") or {})
        verdict["degraded_reasons"] = [*(verdict.get("degraded_reasons") or []), f"review_ledger_unwritten: {exc}"]
        payload["verdict"] = verdict
        return payload, False


def _settle(ctx: ToolContext, wave: _Wave, facts: Dict[str, Any], outcome: Dict[str, Any],
            retention: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build, finish and write the one record; the result is read back from it."""
    from ouroboros.reviewer_slot_config import bind_reviewer_slot_record_id

    drive = ledger_root(ctx)
    record = build_wave_record(facts, surface=wave.request.surface, record_id=wave.record_id, drive_root=drive)
    _finish_record(record, wave, outcome, reuse_key=str(facts.get("reuse_key") or ""), retention=retention)
    # A rejoin settles the SAME record the open operation left pending: the gate's
    # rule (commit_gate: a later settlement raises ``revision``, never a second record).
    payload, durable = _write(drive, record, rejoin=wave.rejoin is not None)
    # The gate's binding: this wave's OWN execution rows (keyed by seat, stamped
    # with the wave's timestamps) take the record id; a later run of a seat by
    # another surface keeps its own.
    with contextlib.suppress(Exception):
        bind_reviewer_slot_record_id(dict(facts.get("slot_executions") or {}), str(payload.get("record_id") or ""))
    durable = settle_attempt(ctx, wave, outcome, payload, facts) is not False and durable
    result = review_result(payload, reused=False)
    result["durable"] = durable
    if wave.request.surface not in SINGLE_QUESTION:
        from ouroboros.review_history_view import queue_review_history_context
        queue_review_history_context(ctx, family="commit", repo_root=wave.root)
    return result


def review_result(record: Dict[str, Any], *, reused: bool) -> Dict[str, Any]:
    """The tool's answer (contract item 3), read from one ledger record."""
    verdict = dict(record.get("verdict") or {})
    checklist = dict((record.get("brief") or {}).get("checklist") or {})
    return {
        "record_id": str(record.get("record_id") or ""), "state": str(record.get("state") or ""),
        "revision": record.get("revision", 1),
        "review_reference": {"record_id": str(record.get("record_id") or ""),
                             "task_id": str(record.get("task_id") or ""),
                             "surface": record.get("surface") if record.get("surface") in SINGLE_QUESTION else "change"},
        "aggregate": str(verdict.get("aggregate") or ""), "per_question": dict(verdict.get("per_question") or {}),
        "quorum": dict(verdict.get("quorum") or {}), "panel": dict(record.get("panel") or {}),
        "rows": list(record.get("rows") or []), "subject": dict(record.get("subject") or {}),
        "checklist": {"layer": str(checklist.get("layer") or "core"),
                      "body_fact": str(checklist.get("body_fact") or "unknown"),
                      "how": str(checklist.get("how") or "unknown"),
                      "treat_as_body": bool(checklist.get("treat_as_body"))},
        "tests": dict(record.get("tests") or {"policy": "NOT_RUN", "result": "unknown"}),
        "cost": {"usd": 0.0, "unknown": False} if reused else dict(record.get("cost") or {}),
        "reused": reused,
        "enforcement": str(record.get("enforcement") or ""),
        "enforcement_blocks": bool(record.get("enforcement_blocks")),
        "findings": {name: list(verdict.get(name) or [])
                     for name in ("critical_findings", "advisory_findings", "additional_findings")},
        "degraded_reasons": list(verdict.get("degraded_reasons") or []),
        "dispatch_refusal": record.get("dispatch_refusal"),
    }


def _cycles_exhausted(ctx: ToolContext, wave: _Wave) -> Optional[Dict[str, Any]]:
    """The shared per-task-tree ceiling of (this root, this kind of review: ``wave_tool_name``); a skill
    wave outside any task (the manual lane) meets the same number per content version (owner Q23)."""
    from ouroboros.tools.commit_gate import check_review_cycles_ceiling

    if wave.request.surface == SKILL and not wave.root_task_id:
        from ouroboros.tools.review_skill import manual_lane_ceiling

        return manual_lane_ceiling(ctx, wave)
    probe = SimpleNamespace(**{**vars(ctx), "repo_dir": wave.root,
                               "_current_review_tool_name": wave_tool_name(wave.request)})
    return check_review_cycles_ceiling(probe, root_task_id=wave.root_task_id)


def _refuse_exhausted(ctx: ToolContext, wave: _Wave, exhausted: Dict[str, Any]) -> Dict[str, Any]:
    from ouroboros.review_cycles import emit_review_cycles_exhausted

    emit_review_cycles_exhausted(
        getattr(ctx, "event_queue", None), ctx.drive_root, surface=wave_tool_name(wave.request),
        task_id=str(getattr(ctx, "task_id", "") or ""), cycles_paid=int(exhausted["cycles_paid"]),
        cap=int(exhausted["cap"]), enforcement=wave.enforcement, root_task_id=wave.root_task_id,
        fingerprint=str(wave.frozen.diff_sha), root=str(wave.root))
    outcome = {"blocked": True, "block_reason": "review_cycles_exhausted",
               "dispatch_refusal": {"kind": "review_cycles_exhausted", "message": str(exhausted["message"])}}
    result = _settle(ctx, wave, wave_facts(ctx, wave, outcome=outcome, forensic={}), outcome)
    result["message"] = str(exhausted["message"])
    return result


# --- The system review (``/review``) ----------------------------------------------

def system_review_row(reviewer: str = "", effort: str = "") -> Any:
    """``/review``'s one executor (decision 3A): an enabled catalog row named by id or
    handle, pool member or not, else the direct Main row; ``effort`` is the call's
    order (a compound route keeps its own)."""
    from ouroboros import reviewer_slot_config as slots

    if not reviewer:
        row = slots.main_review_row()  # Main's own failure (an invalid pin document) is not a catalog refusal
    else:
        try:
            row = slots.catalog_review_row(None, reviewer)
        except ValueError as exc:
            raise ReviewChangeArgumentError(f"reviewer {reviewer!r} is not an enabled catalog row ({exc})") from exc
    return (slots.row_at_effort_order(row, effort) or row) if effort else row


def _as_report(record: Any, drive: Any, text: str) -> None:
    """The system seat answers ``report``, outside the verdict questions: the verdict
    is recomputed over no question (never PASS) and the report is retained as the
    seat's answer."""
    from ouroboros.review_ledger import retain_text_source

    for seat in record.rows:
        seat["parts"], seat["parts_answered"] = [REPORT_PART], [REPORT_PART] if seat["parts_answered"] else []
        if text:
            seat["source_refs"].append(retain_text_source(drive, record.task_id, record_id=record.record_id,
                                                          seat_id=seat["seat_id"], role="response", part=REPORT_PART,
                                                          text=text))
    verdict = reduce_verdict(record.rows, quorum_required=None, gate_blocked=False, gate_reason="",
                             dispatch_refusal=record.dispatch_refusal, pending=False)
    verdict["per_row"] = {seat["seat_id"]: REPORT_PART if seat["parts_answered"] else row_verdict(seat)
                          for seat in record.rows}
    record.verdict = {**dict(record.verdict or {}), **verdict, "report": {"delivered": bool(text), "chars": len(text)}}
    record.brief["parts"] = []


def _system_tree(system: pathlib.Path) -> Tuple[str, str]:
    """The live system tree as the reviewer finds it (``worktree_snapshot_tree``: tracked
    edits and untracked files over HEAD) and HEAD itself, ``""`` where git cannot say."""
    tree = ""
    with contextlib.suppress(Exception):
        from supervisor.update_candidate import worktree_snapshot_tree

        tree, _error = worktree_snapshot_tree("HEAD", cwd=str(system))
    return str(tree or ""), _git_line(system, "rev-parse", "HEAD")


def system_seat_plan(row: Any) -> Dict[str, Any]:
    """The system seat in the wave's shared ``structured["rows"]`` contract (the one
    ``parallel_review._describe_review_wave`` and the gate write): route, effort, session
    target and profile, catalog id, a retrieving delivery — so the record names the row
    actually chosen, and names it even when the review is refused before any send."""
    from ouroboros.reviewer_slot_config import row_effort

    return {"slot_id": row.slot_id, "model": row.target_id, "route": row.kind,
            "effort": row_effort(row), "session_target": row.session_target,
            "session_profile": row.profile_id, "subagent_id": row.subagent_id,
            "retrieves": True, "parts": [PART_CHANGE], "additional": False, "brief_sha": ""}


def run_system_review(ctx: ToolContext, *, reviewer: str = "", effort: str = "", goal: str = "",
                      author_questions: Sequence[str] = (), reason: str = "", llm: Any = None,
                      emit_progress: Any = None, deadline_at: str = "") -> Dict[str, Any]:
    """``/review`` = ``review_change(subject=system, surface=system)``: one seat (``system_review_row``)
    reads the whole system against BIBLE.md through ``deep_self_review`` (retrieving delivery, the
    memory whitelist inline) and writes a report, kept as that seat's answer in one ``surface=system``
    record. ``goal``, ``author_questions`` and ``reason`` are the call's own ask beside the standing
    questionnaire (``deep_self_review.SystemReviewAsk``): one typed object reaches the reviewer's brief
    and the record. The subject is the live tree snapshotted BEFORE the reviewer reads (a tree that
    moved under the review is disclosed as a degraded reason). The report becomes
    ``memory/deep_review.md`` unless the review failed. Returns the record's result plus ``report``
    and ``usage``; ``BudgetExceeded`` propagates."""
    from dataclasses import replace

    from ouroboros.deep_self_review import SystemReviewAsk, run_deep_self_review
    from ouroboros.review_records import new_review_wave_id, review_failure_facts
    from ouroboros.usage_accounting import UsageScope, current_usage_scope, usage_scope

    row = system_review_row(reviewer, effort)
    ask = SystemReviewAsk(goal=str(goal or ""), author_questions=tuple(str(q) for q in author_questions if str(q or "").strip()),
                          reason=str(reason or ""))
    system, drive, record_id = _system_repo(ctx), ledger_root(ctx), new_record_id()
    task_id, started = str(getattr(ctx, "task_id", "") or ""), utc_now_iso()
    if llm is None:
        from ouroboros.llm import LLMClient

        llm = LLMClient()
    tree, head = _system_tree(system)  # the tree the reviewer is about to read
    scope = current_usage_scope() or UsageScope()
    wave = str(getattr(scope, "review_wave_id", "") or "") or new_review_wave_id()
    with usage_scope(replace(scope, review_wave_id=wave)):  # the record and its sends share one round
        report, usage = run_deep_self_review(
            system, pathlib.Path(ctx.drive_root), llm,
            emit_progress or getattr(ctx, "emit_progress_fn", None) or (lambda _text: None),
            task_id=task_id, deadline_at=deadline_at, slot=row, ask=ask, usage_ctx=ctx)
    usage = dict(usage or {})
    failed = str(usage.get("execution_status") or "") == "infra_failed"
    refused = failed and str(usage.get("reason_code") or "") == "deep_self_review_unavailable"
    cost = usage.get("cost")
    fact = body_fact(system, system_repo=system, data_dir=drive, treat_as_body=False)
    degraded = [f"system_review_failed: {usage.get('reason_code')}"] if failed and not refused else []
    after = _system_tree(system)
    if after != (tree, head):
        degraded.append(f"system_tree_moved_during_review: tree {tree[:12]}->{after[0][:12]} head {head[:12]}->{after[1][:12]}")
    record = build_wave_record({
        "task_id": task_id, "root_task_id": str(getattr(ctx, "root_task_id", "") or task_id),
        "review_wave_id": wave,
        "subject": {"root_kind": "system_repo", "root": str(system), "kind": "system", "base": head, "head": head,
                    "tree_sha": tree, "diff_sha": "", "checkout": ""},
        "goal": ask.effective_goal, "author_questions": list(ask.author_questions),
        "layer": layer_for(fact), "body_fact": str(fact.body), "body_how": str(fact.how),
        "enforcement": str(get_review_enforcement() or ""),
        "structured": {"started_ts": started, "rows": [system_seat_plan(row)]},
        "triad_raw": [] if refused else [{
            "slot_id": row.slot_id, "status": "error" if failed else "responded", "raw_text": "" if failed else report,
            "model_id": str(usage.get("resolved_model") or ""), "cost_usd": cost if isinstance(cost, (int, float)) else None,
            "capability_delta": list(usage.get("capability_delta") or []),
            # A failed /review keeps its reason on the row, before the report text can be pruned.
            **(review_failure_facts({"status": "error", "error": report,
                                     "failure_code": str(usage.get("reason_code") or "")}) if failed else {})}],
        "dispatch_refusal": {"kind": "reviewer_unavailable", "message": report} if refused else None,
        "degraded_reasons": degraded,
        "composition": "composed", "composition_reason": ask.reason, "chosen_by": "ouroboros" if reviewer else "owner",
    }, surface="system", record_id=record_id)
    _as_report(record, drive, "" if failed else report)
    # One row by the author's choice (decision 3A) is not a narrowed pool: the reason is
    # recorded as given and its absence is no ``reason_missing`` fact.
    record.panel = {**dict(record.panel or {}), "composition": "composed", "chosen_by": "ouroboros" if reviewer else "owner",
                    "reason": ask.reason, "reason_missing": False, "reviewers_requested": [reviewer] if reviewer else [],
                    "additional": [], "reviewer_effort": {"order": effort}}
    payload, durable = _write(drive, record)
    if not failed:
        target = pathlib.Path(ctx.drive_root) / "memory" / "deep_review.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report, encoding="utf-8")
    result = review_result(payload, reused=False)
    result.update(durable=durable, report=report, usage=usage, report_delivered=not failed)
    return result


def run_review_change(ctx: ToolContext, **args: Any) -> Dict[str, Any]:
    """The tool's runtime entry, shared with the operator script: one subject, at
    most one paid wave, one ledger record. A deterministic call error raises
    ``ReviewChangeArgumentError`` before any reviewer is paid."""
    binding = args.pop("_resolved_binding", None)
    # In-process callers (the commit gate's look, the contributor lane) hand the test
    # execution they established; the tool surface has no such argument.
    supplied_execution = args.pop("_preflight_execution", None)
    public_execution = args.pop("_public_execution", False)
    # The host collecting a running wave it did not start names that wave's recorded round
    # (its retry key): it rejoins the open operation at $0 and never buys another wave.
    rejoin_key = str(args.pop("_rejoin_retry_key", "") or "")
    # The host's delivery step hands the result subject it already froze, so the wave it dispatches or
    # collects is of exactly that material (a recapture could name a later artifact the seats never saw).
    frozen_result = args.pop("_frozen_result", None)
    # ``skill_review`` hands the frozen skill payload it built (``review_skill.skill_subject``): the only
    # way a skill subject enters this operation; with it, the panel its lifecycle already composed by
    # its own source (the Skills page composes as the owner, the tool as the mind).
    frozen_skill = args.pop("_frozen_skill", None)
    supplied_panel = args.pop("_panel", None)
    # Who seated this panel. The tool schema has no such argument; the host floor
    # passes ``host`` and every model call keeps the default.
    chosen_by = str(args.pop("_chosen_by", "") or "ouroboros")
    request = parse_request(args)
    if request.subject == SKILL and frozen_skill is None:
        raise ReviewChangeArgumentError("subject=skill is skill_review's own: call skill_review for a skill payload")
    if request.subject == "system":
        if _assigned_review_root(ctx) is not None:
            raise ReviewChangeArgumentError("review_change is limited to the assigned work; subject=system is unavailable")
        return run_system_review(ctx, reviewer=request.reviewers[0] if request.reviewers else "",
                                 effort=request.reviewer_effort, goal=request.goal,
                                 author_questions=request.author_questions, reason=request.reason)
    governance = _governance_repo(ctx)
    required_root = False
    if request.subject == RESULT:
        from ouroboros.tools.review_result import (
            RESULT_LAYER, refuse_when_off, required_root_wave, result_subject, running_earlier_request,
        )

        refuse_when_off()  # before any subject is built or any seat is paid
        required_root = required_root_wave(ctx)
        result = frozen_result if frozen_result is not None else result_subject(ctx, request, governance_root=governance)
        if frozen_result is None:
            earlier = running_earlier_request(ctx, result)
            # Only to collect: the same unchanged call rejoins that round's open operation at $0; any other call
            # (another brief, a lost operation) reviews the request as it is now.
            if earlier is not result and pending_round_attempt(
                    ctx, root=pathlib.Path(earlier.spec.root), tool_name=wave_tool_name(request),
                    retry_key=review_retry_key(earlier, round_sha=_round_sha(request, earlier))) is not None:
                result = earlier
        root = pathlib.Path(result.spec.root)
        fact = body_fact(root, system_repo=governance, data_dir=ledger_root(ctx), treat_as_body=False)
        layer, spec = RESULT_LAYER, result.spec
    elif request.subject == SKILL:
        from ouroboros.review_body_fact import BodyFact
        from ouroboros.tools.review_skill import SKILL_LAYER

        result, root, spec = frozen_skill, pathlib.Path(frozen_skill.spec.root), frozen_skill.spec
        fact = BodyFact(body="false", how="skill_payload", detail="a skill payload is never the body")
        layer = SKILL_LAYER
    else:
        root_kind, root = resolve_review_root(ctx, binding, request)
        _verify_revisions(root, request)
        fact = body_fact(root, system_repo=governance, data_dir=ledger_root(ctx), treat_as_body=request.treat_as_body)
        layer = layer_for(fact)
        spec = ReviewSubjectSpec(root_kind=root_kind, root=str(root), kind=request.subject, base=request.base,
                                 head=request.head, governance_root=str(governance), surface=request.surface,
                                 body_fact=str(fact.body), body_how=str(fact.how), layer=layer)
    # The gate's own subject (the body's staged index against HEAD) is read on its live
    # root exactly as the gate reads it unless its interactive author may leave the wave
    # on an addressed message (``_read_subject``); every other subject is materialized in an
    # isolated checkout, where every delivery reads the frozen tree, at the path of this
    # TASK's custody of its ROUND (identity c plus the task that keys attempt rows): a
    # rerun of the round reads the checkout its open operation still may, another
    # task's wave of the round never removes it, and open custody after the wave keeps
    # it (review_subject.isolated_checkout).
    retention: Dict[str, Any] = {}
    task_id = str(getattr(ctx, "task_id", "") or "")
    unfrozen: List[str] = []
    reading = (contextlib.nullcontext(result) if request.subject in SINGLE_QUESTION
               else _read_subject(ctx, spec, request, retention, task_id, unfrozen))
    with reading as frozen:
        if not str(getattr(frozen, "diff_text", "") or "").strip():
            raise ReviewChangeArgumentError(f"subject={request.subject} of {root} has no change to review")
        # Locate an already-paid round before consulting mutable pool settings.
        # An absent or changed row cannot substitute a new operation for this one.
        rejoin = pending_round_attempt(ctx, root=root, tool_name=wave_tool_name(request),
            retry_key=rejoin_key or review_retry_key(frozen, round_sha=_round_sha(request, frozen)))
        if rejoin_key and rejoin is None:
            raise ReviewChangeArgumentError("the running review named for collection has no open operation left")
        if rejoin is not None:
            from ouroboros.tools.review_change_custody import recorded_change_panel
            panel = recorded_change_panel(ctx, rejoin)
        elif supplied_panel is not None:
            panel = single_question_panel(supplied_panel, request.surface)
        else:
            panel = compose_panel(request, body=layer == "body", required_root=required_root, chosen_by=chosen_by)
        if rejoin is None and public_execution:
            from ouroboros.review_waiting import bind_panel
            bind_panel(ctx, panel)
        with _panel_in_force(panel):
            wave = dataclasses.replace(_prepare_wave(ctx, request, frozen, panel, root=root, fact=fact, layer=layer,
                                                     rejoin=rejoin),
                                       execution=bind_execution_exhibit(supplied_execution, frozen,
                                                                        public=public_execution))
            if rejoin is not None:
                record_id = str(rejoin.review_record_id or wave.record_id)
                wave = dataclasses.replace(wave, rejoin=rejoin, record_id=record_id,
                                           execution=_dispatched_execution(ctx, record_id, rejoin),
                                           retry_key=rejoin_key or wave.retry_key)
            else:
                prior = reuse_or_none(ledger_root(ctx), wave.reuse_key, questions=request.author_questions)
                if prior is not None:
                    return review_result(prior["record"], reused=True)
                exhausted = _cycles_exhausted(ctx, wave)
                if exhausted is not None:
                    return _refuse_exhausted(ctx, wave, exhausted)
            with _wave_context(ctx, wave):
                outcome = _dispatch(ctx, wave)
                forensic = _forensic(ctx)
                forensic["degraded_reasons"] = [*forensic["degraded_reasons"], *unfrozen]
                retention.update(_checkout_custody(ctx, forensic))
                from ouroboros import review_pause

                kind = request.surface if request.surface in SINGLE_QUESTION else "change"
                settled = _settle(ctx, wave, wave_facts(ctx, wave, outcome=outcome, forensic=forensic), outcome, retention)
                left = (review_pause.watch_left_review(ctx, surface=kind, retry_key=wave.retry_key,
                        record_id=wave.record_id, reissue="review_change(review_reference=the original reference)")
                        if settled.get("durable") else None)
            if settled.get("state") == "pending" and left is None:
                # Nobody waits for these seats any more: their record settles once they finish
                # (at the latest when this task ended), whether or not the call is re-issued.
                review_pause.owe_unwaited_review(ctx, surface=kind, retry_key=wave.retry_key,
                                                 record_id=str(settled.get("record_id") or wave.record_id))
            return settled


@contextlib.contextmanager
def _read_subject(ctx: ToolContext, spec: Any, request: ReviewChangeRequest, retention: Dict[str, Any],
                  task_id: str, unfrozen: List[str]) -> Iterator[Any]:
    """Every subject but the gate's own is read in this round's isolated checkout. A
    receptive author (``review_pause``) may leave the wave, so its gate subject is read
    there too; a checkout that cannot be prepared keeps the live synchronous read, disclosed."""
    from ouroboros import review_pause

    def isolated() -> Any:
        return isolated_checkout(ctx, spec, retain=lambda: retention, token=lambda identity: checkout_token(
            review_retry_key(identity, round_sha=_round_sha(request, identity)), task_id=task_id))
    if not is_gate_subject(spec):
        with isolated() as frozen:
            yield frozen
        return
    if review_pause.receptive_active():
        with contextlib.ExitStack() as stack:
            try:
                frozen = stack.enter_context(isolated())
            except Exception as exc:
                unfrozen.append(f"receptive_wait_unavailable: the reviewed subject could not be frozen "
                                f"({type(exc).__name__}: {exc}); this wave kept the synchronous wait on the live index")
            else:
                yield frozen
                return
    with review_pause.receptive_author(None):
        yield freeze_subject(ctx, spec)


def _handle_new_review_change(
    ctx: ToolContext, root: str = "active_workspace", workspace_root: str = "", subject: str = "",
    base: str = "", head: str = "", surface: str = SURFACE, goal: str = "", scope: str = "",
    author_questions: Optional[List[str]] = None, reviewers: Optional[List[str]] = None, reason: str = "",
    coupling_only: Optional[List[str]] = None, reviewer_effort: str = "", review_rebuttal: str = "",
    treat_as_body: bool = False, answer: str = "", since_review: str = "", _resolved_binding: Any = None,
    wake_on: Any = None,
) -> str:
    from ouroboros import review_pause

    ctx._left_review_wait = None  # a new review act is the author's explicit next choice
    if str(subject or "").strip() == SKILL or str(surface or "").strip() == SKILL:
        return argument_refusal(ctx, "TOOL_ARG_ERROR (review_change)",
                                ["subject=skill is skill_review's own: call skill_review(skill=...) for a skill payload"],
                                effect="No reviewer was dispatched.")
    try:
        from ouroboros.preflight_receipts import execution_exhibit
        exhibit = execution_exhibit(ctx)
        # The tool's interactive author may leave the wave on an addressed message (the
        # operator script's synchronous entry binds nothing).
        from ouroboros.review_waiting import select_helpers
        state = select_helpers(ctx, wake_on)
        if str(subject or "").strip() == "system" and any((wake_on or {}).values()):
            raise ValueError("wake_on requires retained review custody; subject=system uses a synchronous report wait")
        with review_pause.receptive_author(review_pause.receptive_binding(ctx, state=state)):
            result = run_review_change(
                ctx, root=root, workspace_root=workspace_root, subject=subject, base=base, head=head, surface=surface,
                goal=goal, scope=scope, author_questions=author_questions, reviewers=reviewers, reason=reason,
                coupling_only=coupling_only, reviewer_effort=reviewer_effort, review_rebuttal=review_rebuttal,
                treat_as_body=treat_as_body, answer=answer, since_review=since_review,
                _resolved_binding=_resolved_binding, _preflight_execution=exhibit if exhibit.get("status") == "passed" or exhibit.get("observed") else None,
                _public_execution=True)
    except ValueError as exc:
        return argument_refusal(ctx, "TOOL_ARG_ERROR (review_change)", [str(exc)], effect="No reviewer was dispatched.")
    result.pop("usage", None)
    if (getattr(ctx, "_left_review_wait", None) or {}).get("surface") in {"change", RESULT, SKILL}:
        event = ("A selected outcome or wake time was ready" if any(row.get("reason") in {"helper_result", "review_result", "wake_at"} for row in
                 getattr(ctx, "_review_detached_operations", []) or []) else "An addressed message reached you")
        result["message"] = (event + " while the reviewers ran: this call returned and their "
                             "paid work continues separately; a [Review settled] notice follows the whole wave. "
                             + str(result.get("message") or ""))
    text = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    return f"{result['message']}\n\n{text}" if result.get("message") else text


def _handle_review_change(ctx: ToolContext, review_reference: Any = None, *,
                          _resolved_binding: Any = None, **args: Any) -> str:
    from ouroboros.review_waiting import publish_delivery
    from ouroboros.tools.review_collection import collect_change

    if review_reference is None:
        text = _handle_new_review_change(ctx, _resolved_binding=_resolved_binding, **args)
        # The record id is also returned to direct callers; the metadata is for the loop.
        try:
            payload = json.loads(text[text.index("{"):])
        except (ValueError, TypeError):
            return text
    else:
        try:
            if args:
                raise ValueError("review_reference cannot be combined with subject, brief or panel arguments")
            payload = collect_change(ctx, review_reference)
            text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        except (ValueError, LookupError) as exc:
            return argument_refusal(ctx, "TOOL_ARG_ERROR (review_change)", [str(exc)],
                                    effect="No reviewer was dispatched.")
    record = load_record(ledger_root(ctx), str(payload.get("record_id") or "")) if payload.get("record_id") else None
    return publish_delivery(ctx, text, record) if record else text


def _review_change_tool_timeout_sec() -> float:
    # The plan_task envelope: a settlement bound covering an API transport read or
    # a delegated session's operation window, never a cognition cutoff.
    return (max(float(get_llm_transport_read_timeout_sec() + get_finalization_grace_sec()),
                operation_window_sec(get_task_abs_ceiling_sec()))
            + get_finalization_grace_sec())


_DESCRIPTION = (
    "Review a change WITHOUT committing it: one paid reviewer wave over a subject in any registered repository, "
    "written to the review ledger. commit_reviewed stays the only landing in the system repository; use "
    "review_change by judgment for any other root (it never starts on its own and never runs tests: "
    "tests.policy=NOT_RUN). Subjects: base..head (commits), index (staged) or worktree (live) against base "
    "(default HEAD), frozen and read in an isolated checkout of its tree (only a synchronous caller reads the body's "
    "own staged index live). An addressed message while the wave runs returns its pending outcome: the reviewers "
    "continue, a [Review settled] notice follows, and review_reference collects the original subject at $0 even after edits. "
    "Rules follow the subject: the body is read against the body "
    "layer, anything else against the universal core checklist. Panel: Blocking own-body reviews widen to the whole "
    "pool; Advisory may narrow with a reason (missing reason is a typed $0 refusal); Cyber Pro composes freely. "
    "Other repositories allow composed panels. Omitted means the whole pool; an enabled unticked row is an added "
    "critic. surface=preflight seats exactly the ONE enabled catalog row you name (pool member or "
    "not): an optional early look, never the commit panel. subject=system with surface=system is /review: one row "
    "(default: the Main model) reads the whole system against BIBLE.md and returns a report. subject=result "
    "(surface=result) reviews THIS task's result instead of a change: the reply (answer, default your latest reply), "
    "the files and effects the task produced, and what was asked; every seat answers one question, whether the "
    "result does what was asked, judged on the result, and on the method only where the owner's words or "
    "explicit choice fixed it. On a Required root, Blocking widens a "
    "named subset to every ticked reviewer; Advisory names ticked reviewers and needs a reason when narrowed "
    "(missing reason is a typed $0 refusal); if you did not review before delivery the host seats the whole pool. "
    "Auto composes the rows you name in every strictness. Cyber Pro composes freely and never holds. A child's "
    "own result review composes freely. Omitted reviewers mean the whole pool. Under Auto ask when another mind's "
    "check could change what you deliver, or when your human asks for a careful check. It is unavailable when "
    "task review is Off, and it has its own paid-cycle count. The same subject, surface, panel and ask return the "
    "settled record free (reused=true), even after a guide changed; enforcement is recorded and on another "
    "root never locks anything. Returns JSON {record_id, aggregate, per_question, panel, rows, subject, checklist, "
    "tests, cost, reused, findings} (+ report for subject=system)."
)


def _string(description: str, **extra: Any) -> Dict[str, Any]:
    return {"type": "string", "default": "", "description": description, **extra}


def _names(description: str) -> Dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}, "default": [], "description": description}


def get_tools() -> List[ToolEntry]:
    from ouroboros.review_waiting import wake_on_schema
    properties = {
        "wake_on": wake_on_schema(),
        "review_reference": {"type": "object", "description": "Collect the original paid review at $0; supply no other arguments."},
        "root": _string("The bound repository: the active workspace or the system repository.",
                        enum=list(ROOTS), default="active_workspace"),
        "workspace_root": _string("A registered folder to review instead (schedule_subagent's workspace_root policy)."),
        "subject": {"type": "string", "enum": list(PUBLIC_SUBJECT_KINDS),
                    "description": ("base..head = commits between two revisions; index = staged; worktree = live; "
                                    "system = the whole installed system (surface=system only); result = this "
                                    "task's result against what was asked.")},
        "base": _string("Base revision (required for base..head; index/worktree default HEAD)."),
        "head": _string("Head revision (base..head only)."),
        "surface": _string("change = the panel review; preflight = one named row's early look; system = /review; "
                           "result = goes with subject=result (implied by it).", enum=list(PUBLIC_SURFACES), default=SURFACE),
        "answer": _string("subject=result only: the reply under review. Omitted = your latest reply in this task."),
        "goal": _string("What the change is meant to achieve."),
        "scope": _string("What the change deliberately covers."),
        "author_questions": _names("Your own questions to the reviewers, passed verbatim."),
        "reviewers": _names("Pool seats (counted) or enabled catalog rows outside the pool (added critics), by id "
                            "or handle (see the panel rule)."),
        "reason": _string("Why this panel; required for Advisory narrowing, recorded with it."),
        "coupling_only": _names("Extra critics for the coupling question only; outside the quorum."),
        "reviewer_effort": {"type": "string", "enum": list(EFFORT_SCALE), "description": (
            "Panel strength for THIS wave; outranks each seat's effort (a compound route keeps its own). "
            "Omitted = the owner's settings.")},
        "review_rebuttal": _string("Your answer to the previous wave's findings; buys a new wave."),
        "treat_as_body": {"type": "boolean", "default": False, "description": (
            "Raise a root whose body fact is unknown (git cannot place it: no remote, no copy binding) to the "
            "body layer. A recognized body and a recognized foreign root are unchanged; the record notes the raise.")},
        "since_review": _string(
            "A review_record_id of an earlier change review of this same repository (another task or copy of "
            "it) to compare with: seats are shown what changed since that review's subject, its verdict only as a "
            "label of those earlier bytes; no PASS, finding or obligation carries over. An unknown, foreign or "
            "unreadable record is a stated comparison gap. Omitted: the reviews this task already bound on this "
            "checkout are compared."),
    }
    schema = {"name": "review_change", "description": _DESCRIPTION,
              "parameters": {"type": "object", "properties": properties, "required": []}}
    return [ToolEntry("review_change", schema, _handle_review_change, timeout_sec=_review_change_tool_timeout_sec())]
