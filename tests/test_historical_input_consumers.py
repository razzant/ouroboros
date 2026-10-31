"""Historical premises reach the terminal package through the existing writer.

The author loop captures the inputs a turn was authored from as one immutable
historical source; the terminal writer stores that exhibit beside the task inputs,
never inside the owner corpus. The acceptance packet and its source-first reader
that used to consume it are gone with the task-acceptance machine. Only the author
model transport returns scripted responses; none of this is evidence of a real
provider or model comprehension.
"""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

from ouroboros.artifacts import read_actor_source_bytes
from tests import test_main_authored_context as context_fixtures

main_loop = context_fixtures.main_loop


_OLD = "HISTORICAL-AUDIENCE-EXECUTIVE-74a2"
_TODAY = "CURRENT-AUDIENCE-PUBLIC-91d8"


def test_direct_author_loop_capture_and_terminal_package_share_source(main_loop, monkeypatch):
    """Actual author capture → live exhibit → terminal writer, one retained source.

    The post-task dispatcher is only recorded: it receives no review evidence of its
    own (no stage reads it). Only the author model transport returns scripted responses.
    """
    from ouroboros import agent_task_pipeline as pipeline
    from ouroboros.context_input_selection import historical_inputs_exhibit
    from tests.test_historical_inputs import _input

    f = main_loop
    f.ctx.current_chat_id = 12
    f.ctx.task_metadata = {"source": "web_chat", "_is_direct_chat": True}
    f.ctx.task_contract = {"objective": "Revise the feature list."}
    # The premise is beyond the preview bound, so only an actual source read can
    # deliver it to a later reader. No predecessor or room lookup supplies it.
    _input(f, "Captured earlier room context. " * 100 + _OLD, "Revise the feature list.")
    answer, usage, trace = f.run([{"content": "The feature list is revised."}])
    historical = historical_inputs_exhibit(f.ctx)
    assert historical["status"] == "captured" and len(historical["anchors"]) == 1
    ref = historical["anchors"][0]["source_ref"]
    assert not f.ctx.task_contract.get("predecessor_authority")
    assert _OLD not in historical["anchors"][0]["preview"]
    original_raw = read_actor_source_bytes(f.ctx.drive_root, f.ctx.task_id, ref)
    assert _OLD in original_raw.decode("utf-8")

    dispatched = []
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async",
                        lambda env, task, post_usage, llm_trace, _logs, **kwargs: dispatched.append(sorted(kwargs)))
    task = {"id": f.ctx.task_id, "type": "task", "chat_id": 12, "text": "Revise the feature list.",
            "_is_direct_chat": True, "metadata": copy.deepcopy(f.ctx.task_metadata),
            "drive_root": str(f.ctx.drive_root), "task_contract": f.ctx.task_contract,
            "workspace_root": str(f.ctx.repo_dir), "workspace_mode": "external"}
    (f.ctx.repo_dir / "current-author-input.json").write_text(_TODAY, encoding="utf-8")
    pipeline.emit_task_results(SimpleNamespace(drive_root=f.ctx.drive_root, repo_dir=f.ctx.repo_dir),
        None, None, [], task, answer, usage, trace, 0, f.ctx.drive_root / "logs", ctx=f.ctx)
    saved = pipeline.load_task_result(f.ctx.drive_root, f.ctx.task_id)["review_evidence"]
    assert len(dispatched) == 1 and "review_evidence" not in dispatched[0]
    assert saved["historical_author_inputs"] == historical
    assert "historical_author_inputs" not in saved["task_inputs"]
    assert _OLD not in json.dumps(saved["task_inputs"])
    assert read_actor_source_bytes(f.ctx.drive_root, f.ctx.task_id, ref) == original_raw
