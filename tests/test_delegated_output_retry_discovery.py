"""R3: retained outputs stay discoverable through ordinary ``get_task_result`` after a retry.

``get_task_result(A)`` answers with A's EFFECTIVE result, which retry successor B
replaces, while ``delegated_output_sha256`` stays pinned to the PHYSICAL task whose
receipts publish the source. A detached reader starts at the public summary and
follows only the recipe it returns — never a digest taken from a staging fixture —
for A's late-published source, B's own source, both, and after B's staging drive is
removed. The pin is not loosened: each digest under the other task stays unpublished.
"""

from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

import pytest

from ouroboros import delegate_custody as custody
from ouroboros import delegate_terminal
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.task_status import load_effective_task_result
from ouroboros.tools.registry import ToolContext
from tests.test_delegate_late_output_publication import discharged_retry
from tests.test_delegate_retry_output import (
    TerminalGateway,
    bind_gateway,
    call,
    durable_run,
    receipts,
    wait_result,
)
from tests.test_delegate_retry_output import (
    isolated_retry_state as isolated_retry_state,  # autouse fixtures of the retry suites
)
from tests.test_delegate_retry_output import (
    reader_coverage as reader_coverage,
)



def _summary(reader, task_id):
    text = call(reader, "get_task_result", task_id=task_id).text
    return json.loads(text.split("[SUBTASK_OUTCOME]\n", 1)[1].split("\n[/SUBTASK_OUTCOME]", 1)[0])


def _advertised(summary):
    view = summary.get("delegated_custody") or {}
    rows = [*view.get("terminal_runs", []), *view.get("predecessor_terminal_runs", [])]
    return {row["run_id"]: row["staged_output"] for row in rows if isinstance(row.get("staged_output"), dict)}


def _source(reader, recipe_or_args, length=80):
    result = call(reader, "get_task_result", **recipe_or_args, source_start_char=0, source_end_char=length)
    assert result.status == "ok", result.text
    return json.loads(result.text)["delegated_output_source"]


@pytest.mark.parametrize(("published", "cleanup"), [
    pytest.param("a_only", False, id='a_only-drive-kept'),
    pytest.param("b_only", True, id='b_only-drive-removed'),
    pytest.param("both", False, id='both-drive-kept'),
])
def test_ordinary_discovery_follows_each_publishing_physical_task(tmp_path, monkeypatch, published, cleanup):
    ctx, held, _before = discharged_retry(tmp_path, monkeypatch)  # A failed + discharged; B is the live retry
    data = custody.custody_root(ctx)
    expected = {}
    if published in ("a_only", "both"):
        # B reads A's run: staged into B's store, published late on physical A.
        expected[held.run_id] = ("retry-a", wait_result(ctx, held.run_id)["output_delivery"]["artifact"])
    if published in ("b_only", "both"):
        own = durable_run(ctx, owner="retry-b", run_id="run-b")
        bind_gateway(monkeypatch, TerminalGateway(output="the successor's own work " * 8000))
        expected[own.run_id] = ("retry-b", wait_result(ctx, own.run_id)["output_delivery"]["artifact"])
        write_task_result(data, "retry-b", "completed", result="done")
        delegate_terminal.record_terminal_reconciliation(
            data, "retry-b", delegate_terminal.terminal_reconcile_task(data, "retry-b"))
        stored = load_task_result(data, "retry-b")["delegate_terminal_reconciliation"]["terminal_runs"]
        assert any(row.get("staged_output") for row in stored if row["run_id"] == own.run_id)
    if cleanup:
        shutil.rmtree(ctx.drive_root)  # B's split execution drive is gone
    custody._CUSTODY.clear()
    reader = ToolContext(repo_dir=ctx.repo_dir, drive_root=data, task_id="physical-reader")
    assert load_effective_task_result(data, "retry-a")["task_id"] == "retry-b", "the effective view is B's"

    summary = _summary(reader, "retry-a")
    advertised = _advertised(summary)
    assert set(advertised) == set(expected)
    view = summary["delegated_custody"]
    assert {row["run_id"] for row in view.get("predecessor_terminal_runs", [])} == (
        {held.run_id} if published != "b_only" else set()), "A's receipt stays under its own lineage hop"
    for run_id, (publisher, artifact) in expected.items():
        staged = advertised[run_id]
        assert staged["sha256"] == artifact["sha256"] and staged["task_id"] == "retry-b", "B's store stages both"
        recipe = staged["source_read"]
        assert recipe == {"tool": "get_task_result", "arguments": {
            "task_id": publisher, "delegated_output_sha256": artifact["sha256"]}}
        page = _source(reader, recipe["arguments"])
        assert page["status"] == "available" and page["run_id"] == run_id and page["staging_task_id"] == "retry-b"
        assert page["text"] == Path(artifact["abs_path"]).read_text(encoding="utf-8")[:80]
        other = "retry-b" if publisher == "retry-a" else "retry-a"
        pinned = _source(reader, {"task_id": other, "delegated_output_sha256": artifact["sha256"]})
        assert pinned["reason"] == "source_not_published", "the physical pin is not widened"

    # The authority view is the same effective row: the predecessor's receipts ride its hop.
    authority = json.loads(call(reader, "get_task_result", task_id="retry-a", include_authority=True).text)
    hop = authority["authority"]["retry_lineage"][0]
    assert hop["task_id"] == "retry-a" and hop["retry_task_id"] == "retry-b"
    assert [row["run_id"] for row in hop.get("terminal_runs", [])] == (
        [held.run_id] if published != "b_only" else [])
    assert not receipts(ctx), "discovery and selector reads never acknowledge"


def test_a_successor_without_retained_outputs_adds_no_lineage_receipts(tmp_path, monkeypatch):
    """Rows that publish no retained source keep the effective view's prior shape."""
    ctx, _held, _before = discharged_retry(tmp_path, monkeypatch)
    data = custody.custody_root(ctx)
    reader = ToolContext(repo_dir=ctx.repo_dir, drive_root=data, task_id="physical-reader")
    hop = load_effective_task_result(data, "retry-a")["retry_lineage"][0]
    assert "terminal_runs" not in hop
    assert "predecessor_terminal_runs" not in (_summary(reader, "retry-a").get("delegated_custody") or {})


@pytest.mark.parametrize("publisher", ["retry-a", "retry-b"])
def test_omitted_outputs_are_readable_by_following_the_advertised_full_source(tmp_path, monkeypatch, publisher):
    """A ten-row summary must lead to the omitted identifiers, not itself again."""
    ctx, _held, _before = discharged_retry(tmp_path, monkeypatch)
    data = custody.custody_root(ctx)
    expected = {}
    for number in range(12):
        run = durable_run(ctx, owner=publisher, run_id=f"run-many-{number:02d}")
        artifact = wait_result(ctx, run.run_id)["output_delivery"]["artifact"]
        expected[run.run_id] = artifact["sha256"]
    if publisher == ctx.task_id:
        write_task_result(data, publisher, "completed", result="done")
        delegate_terminal.record_terminal_reconciliation(
            data, publisher, delegate_terminal.terminal_reconcile_task(data, publisher))
    shutil.rmtree(ctx.drive_root)
    custody._CUSTODY.clear()
    reader = ToolContext(repo_dir=ctx.repo_dir, drive_root=data, task_id="physical-reader")
    disclosure = _summary(reader, "retry-a")["delegated_custody"]
    field = "predecessor_terminal_runs" if publisher == "retry-a" else "terminal_runs"
    assert disclosure[field + "_omitted"] == 2
    discovered = {}
    for recipe in disclosure["full_source"]:
        parsed = ast.parse(recipe, mode="eval").body
        assert isinstance(parsed, ast.Call) and parsed.func.id == "get_task_result"
        arguments = {keyword.arg: ast.literal_eval(keyword.value) for keyword in parsed.keywords}
        result = call(reader, "get_task_result", **arguments)
        assert result.status == "ok" and result.text.startswith("{"), "full_source repeated the bounded summary"
        authority = json.loads(result.text)["authority"]
        rows = (authority.get("delegate_terminal_reconciliation") or {}).get("terminal_runs", [])
        rows += [row for hop in authority.get("retry_lineage", []) for row in hop.get("terminal_runs", [])]
        discovered.update({row["run_id"]: row["staged_output"] for row in rows if "staged_output" in row})
    assert set(discovered) == set(expected)
    for run_id, staged in discovered.items():
        assert staged["sha256"] == expected[run_id]
        page = _source(reader, staged["source_read"]["arguments"])
        assert page["status"] == "available" and page["run_id"] == run_id
    assert not receipts(ctx), "discovery never grants a consumption acknowledgement"
