"""Child-only fixture: the real loop in its own interpreter until the parent SIGKILLs it.

Never point its root at an installation. The model and the external effects
are local doubles; the loop, its working-checkpoint writer and the owner
mailbox are real. A hard kill runs no ``finally``, no cleanup and no deferred
ACK, which an in-process exception cannot show.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

TASK_ID = "t-kill"
FIRST_MAIL = "first owner instruction: keep the audit order"
SECOND_MAIL = "second owner instruction: also cite the receipt"
# Each kill point and the calls of its one accepted batch. Two read-only names
# run in the loop's shared pool; any other name forces sequential execution.
BATCHES = {
    "inflight_effect": ("kill_probe", ["effect-1"]),
    "inflight_sequential_batch": ("kill_probe", ["effect-1", "effect-2"]),
    "inflight_parallel_batch": ("read_file", ["effect-1", "effect-2"]),
    "torn_ready_write": ("kill_probe", ["effect-1"]),
}


def record_effect(root: Path, call_id: str) -> None:
    """The external effect a replay would repeat: one durable line per execution."""
    with (root / "effects.log").open("a", encoding="utf-8") as handle:
        handle.write(call_id + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def recorded_effects(root: Path) -> list:
    path = root / "effects.log"
    return path.read_text(encoding="utf-8").split() if path.exists() else []


def _park(root: Path, facts: dict) -> None:
    (root / "killpoint.json").write_text(json.dumps({"pid": os.getpid(), **facts}), encoding="utf-8")
    while True:  # only the parent's SIGKILL ends this interpreter
        time.sleep(60)


def main(root: str, scenario: str) -> None:
    import pytest

    from ouroboros import loop, utils
    from ouroboros import working_checkpoint as wc
    from ouroboros.owner_mailbox import write_owner_message
    from ouroboros.tools.tool_result import ToolResult
    from tests.test_loop_transport_wait import _loop_kwargs
    from tests.test_working_checkpoint import _registry

    base = Path(root)
    drive = base / "drive"
    tool, call_ids = BATCHES[scenario]
    patch = pytest.MonkeyPatch()
    registry = _registry(drive, patch, TASK_ID, 1)
    assert write_owner_message(drive, FIRST_MAIL, TASK_ID, msg_id="mail-1")
    patch.setattr("ouroboros.llm.LLMClient.chat", lambda *a, **k: pytest.fail("no provider call"))
    calls = {"model": 0}

    def dispatch(call, disposition, *, candidate_predicate=None, **kwargs):
        calls["model"] += 1
        if calls["model"] > 1:
            raise AssertionError("the child must be killed before a second model turn returns")
        return {"role": "assistant", "content": "", "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": tool, "arguments": json.dumps({"path": call_id})}}
            for call_id in call_ids]}, 0.0

    def execute_result(name, args):
        call_id = args["path"]
        assert name == tool and call_id in call_ids, (name, args)
        record_effect(base, call_id)
        if scenario.startswith("inflight") and call_id == call_ids[-1]:
            # Die inside the batch only after every earlier call's effect happened.
            while sorted(recorded_effects(base)) != sorted(call_ids):
                time.sleep(0.01)
            _park(base, {"point": scenario, "effects": recorded_effects(base)})
        if scenario == "torn_ready_write":
            # The owner writes while the effect runs; the next round's drain reads it.
            assert write_owner_message(drive, SECOND_MAIL, TASK_ID, msg_id="mail-2")
            return ToolResult(status="ok", code="OK", text="receipt R-17 recorded")
        return ToolResult(status="ok", code="OK", text=f"{call_id} completed")

    real_replace = utils.replace_atomic

    def replace_then_die(src, dst, **kwargs):
        target, temp = Path(dst), Path(src)
        if (scenario == "torn_ready_write" and target.name.startswith(wc.FILE_PREFIX)
                and SECOND_MAIL.encode("utf-8") in temp.read_bytes()):
            # Die between the temp write and os.replace, the temp itself half written.
            data = temp.read_bytes()
            working = json.loads(data)["working"]
            temp.write_bytes(data[: len(data) // 2])
            _park(base, {"point": scenario, "temp": temp.name, "full_bytes": len(data),
                         "boundary": working["boundary"], "seq": working["seq"]})
        return real_replace(src, dst, **kwargs)

    patch.setattr(loop, "_dispatch_round_model", dispatch)
    patch.setattr(registry, "execute_result", execute_result, raising=False)
    patch.setattr(utils, "replace_atomic", replace_then_die)
    kwargs = _loop_kwargs(drive, registry, [])
    kwargs["task_id"] = TASK_ID
    loop.run_llm_loop(**kwargs)
    raise AssertionError("the loop returned; the kill point was never reached")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
