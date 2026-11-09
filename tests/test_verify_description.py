"""The ``verify_and_record`` schema speaks to every task of every install (audit 08.11 item 11).

The rule about where a task's contract comes from has one home, ``prompts/SYSTEM.md`` ("The contract
comes ONLY from the task statement, its materials, and my human"); the protection of a benchmark's
hidden graders is a mechanism (``verify.py`` refuses to probe absolute or traversing artifact paths in
the executor: ``tests/test_verify_artifact_lifecycle.py::test_probe_traversal_is_unavailable_not_probed``).
The schema names what a receipt rests on and no benchmark's files.
"""
from __future__ import annotations

from ouroboros.tools.verify import get_tools


def _description() -> str:
    return next(entry.schema for entry in get_tools() if entry.name == "verify_and_record")["description"]


def test_verify_description_names_the_task_and_the_human_not_a_benchmark():
    description = _description()
    assert ("Verify against the task's own statement and materials, your human's words, installed oracles "
            "and your independent checks.") in description
    for benchmark_word in ("solution.sh", "hidden /tests/", "PUBLIC task", "look up the answer online"):
        assert benchmark_word not in description, benchmark_word
