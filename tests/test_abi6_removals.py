"""ABI-6 removal pins: the three F3.0 P1 removals, proven negatively.

The F3.0 opening train removed three dead surfaces (`03a835b9`, `e94f063d`):
the `_call_llm_with_retry` alias in `ouroboros/loop.py`, and the second
with-children cost rollup `compute_cost_with_children` plus the unused
`format_handoff_message` in `ouroboros/task_status.py`. Each removal was
gated only by the suites of the code that SURVIVED, so every one of those
hooks stays green if an alias comes back — a positive suite cannot notice an
extra name. The ADOPTION row R ABI-6 carries that as a disclosed residual:
"proven by surviving positive suites plus grep-level absence, not by
dedicated negative pins". This file is the missing negative pin.

The assertions are deliberately two-sided. Absence alone is satisfied by
deleting the module, so each retired name is paired with the canonical
surface that replaced it: a resurrection is red, and so is losing the
replacement.
"""

from __future__ import annotations

import importlib
import inspect
import re

import pytest

# (former owner module, retired name, what carries the job now)
RETIRED = (
    ("ouroboros.loop", "_call_llm_with_retry",
     "the public call_llm_with_retry (the one patch point)"),
    ("ouroboros.task_status", "compute_cost_with_children",
     "the producer rollup in agent_task_pipeline/post_task_synthesis, "
     "projected through cost_projection.py"),
    ("ouroboros.task_status", "format_handoff_message",
     "the absorption / wait_tasks projections"),
    ("ouroboros.task_status", "_handoff_snippet",
     "nothing — private helper of the removed formatter"),
    ("ouroboros.task_status", "HANDOFF_SNIPPET_CHARS",
     "nothing — private constant of the removed formatter"),
)


def _names_the_symbol(source: str, name: str) -> bool:
    """Whole-identifier match, so `fake_call_llm_with_retry` in a test double
    or `call_llm_with_retry` itself never counts as the retired spelling."""
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}\b", source) is not None


@pytest.mark.parametrize(("module_name", "retired", "successor"), RETIRED,
                         ids=[f"{m.rsplit('.', 1)[-1]}.{n}" for m, n, _ in RETIRED])
def test_a_retired_p1_name_is_absent_from_its_former_owner(module_name, retired, successor):
    module = importlib.import_module(module_name)

    assert not hasattr(module, retired), (
        f"{module_name}.{retired} is back; the job belongs to {successor}")
    # A re-export binds the attribute too, but a re-introduced *def* under a
    # conditional or inside a class body would not — the source check catches
    # the spelling wherever it reappears in the former owner.
    assert not _names_the_symbol(inspect.getsource(module), retired), (
        f"{module_name} names {retired} again")


def test_the_surviving_llm_retry_surface_is_the_public_one():
    """The alias existed "for source-inspecting/monkeypatched tests"; every
    such test targets the public name, which must therefore still be there."""
    from ouroboros import loop

    assert callable(loop.call_llm_with_retry)


def test_the_one_with_children_cost_rollup_is_the_projection_ssot():
    """`compute_cost_with_children` was the SECOND, diverging rollup. The
    remaining one is the producer path's, read through the projection SSOT."""
    from ouroboros import cost_projection, task_status

    assert callable(cost_projection.live_root_cost_projection)
    assert callable(cost_projection.cost_projection)
    # task_status keeps its own non-rollup readers (the child lookup the
    # removal's own commit preserved), so the module is not simply gone.
    assert callable(task_status.find_child_tasks)
