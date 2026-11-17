"""One-shot facts at the finish door and the answer they returned (leaf of ``loop_delivery``).

The pre-delivery notice, a red check, children's results and a non-blocking result review's outcome each return an
answer once. Like the routeless note they leave the next answer fresh: prose after them is delivered as written,
without a selection only ``finish_task`` could make (on the stand Qwen3.5-9B repeated its report instead, and its
turn never ended). Three rules keep that honest for every model:

* the text the model wrote to the person never disappears: when a later final (prose, a selection or a forced
  answer) replaces the answer the facts returned, that answer is shown as an interim message first, unless the new
  answer is it or contains it whole, and the task's result keeps it apart from the final's bytes for its readers
  outside the room (``DISPLACED_ANSWER_KEY``);
* a silent selection stays silent: prose after the facts inherits it, and only an explicit ``finish_task`` without
  ``silent`` speaks;
* work after the facts (evidence changed) leaves the next prose a fresh answer bound to the new evidence; a selection
  by ``answer_sha256`` names the old answer with its old evidence, so changed evidence returns once and a repeated
  selection then delivers until new work.
Real debts (owner mail, a blocking review, a skill action, handover, plan, services) and an armed episode hold as
before.
"""
from __future__ import annotations

from typing import Any, Dict

FACTS_NOTED_CONTROL = "delivery_facts_noted"
FACTS_NOTED_LINE = "Your next answer is delivered as written, so make it the whole answer."
# After a silent selection: the next prose inherits that silence, so the model is told before it writes (review 8).
FACTS_NOTED_QUIET = " You chose silence, so it stays unspoken; only finish_task without silent speaks."
# Trace key of the answers later finals displaced, in order and without repeats: the result keeps each apart from the
# final's bytes, for consumers outside the room (``task_finalization.build_completion_observations``,
# ``displaced_answer_note``).
DISPLACED_ANSWER_KEY = "displaced_answer"


def _loop():
    from ouroboros import loop

    return loop


def note_delivery_facts(tools: Any, limit_ctx: Any, llm_trace: Dict[str, Any]) -> None:
    """Leave the next answer fresh after one-shot facts; inside an existing control episode arm control as before."""
    candidate = getattr(tools._ctx, "_delivery_candidate", None)
    if candidate is None or candidate.control_episode_seen:
        return _loop()._arm_delivery_control(tools, limit_ctx, llm_trace)
    _loop()._append_or_merge_user_message(limit_ctx.messages,
                                          FACTS_NOTED_LINE + (FACTS_NOTED_QUIET if candidate.quiet else ""))
    candidate.finalization_control, tools._ctx._delivery_control_required = FACTS_NOTED_CONTROL, False
    tools._ctx._completion_pair_appended = tools._ctx._completion_pair_private = True  # no hold notice follows
    _loop()._publish_delivery_candidate(tools, candidate, llm_trace)


def noted(candidate: Any) -> bool:
    """Whether ``candidate`` is the answer one-shot facts returned, its next answer still fresh."""
    return getattr(candidate, "finalization_control", "") == FACTS_NOTED_CONTROL and not candidate.control_episode_seen


def show_replaced_answer(prior: Any, was_noted: bool, current: Any, emit_progress: Any,
                         llm_trace: Dict[str, Any]) -> None:
    """Show the answer the facts returned once, before the final that replaced it. Not shown: the same answer, one the
    new answer contains whole (it is then shown once, as the final), and a silent selection's words."""
    text = str(getattr(prior, "full_text", "") or "")
    if (not was_noted or prior is current or not text.strip() or getattr(prior, "quiet", False)
            or getattr(current, "quiet", False) or text in str(getattr(current, "full_text", "") or "")):
        return
    if text not in llm_trace.setdefault(DISPLACED_ANSWER_KEY, []):
        llm_trace[DISPLACED_ANSWER_KEY].append(text)
    _loop()._emit_round_progress(text, {"content": text}, emit_progress, llm_trace)


def select_answer(tools: Any, ctx: Any, trace: dict, text: str, request: dict, emit_progress: Any) -> Any:
    """Retain an explicit choice and show the facts' displaced answer before it is forgotten."""
    prior = getattr(tools._ctx, "_delivery_candidate", None)
    was_noted = noted(prior)
    by_hash = prior is not None and request.get("answer_sha256") == prior.content_sha256
    evidence = _loop()._delivery_evidence_state(tools, ctx, trace) if by_hash else None
    if by_hash and prior.finalization_control != evidence_notice_control(*evidence):
        candidate = prior  # selection names the old answer, including its old evidence
        candidate.finalization_control = "selected"
        tools._ctx._delivery_control_required = False
    else:  # full bytes, or a repeated selection after the changed evidence was shown
        candidate = _loop()._replace_delivery_candidate(tools, ctx, trace, text, control="selected")
    candidate.quiet = bool(request.get("wake_silent"))
    _loop()._publish_delivery_candidate(tools, candidate, trace)
    if emit_progress is not None:
        show_replaced_answer(prior, was_noted, candidate, emit_progress, trace)
    return candidate


def evidence_notice_control(revision: int, fingerprint: str) -> str:
    """The evidence returned at the exit door; a new work revision withdraws this acknowledgement."""
    return f"evidence_changed:{revision}:{fingerprint}"


def note_shown_review(tools: Any, limit_ctx: Any, llm_trace: Dict[str, Any], shown: list, cause: str) -> None:
    """An outcome shown once is a one-shot fact; a blocking hold is a debt, and nothing is noted for it."""
    from ouroboros.tools.review_result import BLOCKING_REFUSAL_CAUSE, _blocking_enforcement

    if shown and cause != BLOCKING_REFUSAL_CAUSE and not _blocking_enforcement():
        note_delivery_facts(tools, limit_ctx, llm_trace)
