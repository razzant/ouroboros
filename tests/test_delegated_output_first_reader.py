"""#1859: the FIRST advertised reader, including old receipts, is actually usable."""
from __future__ import annotations

import json
import shutil

import pytest

from ouroboros.artifacts import task_artifact_dir_path
from ouroboros.delegate_output import retain_terminal_bytes
from ouroboros.owner_stop_receipt import _prompt_run
from ouroboros.task_results import load_task_result, write_task_result
from tests import test_delegated_run_discovery as discovery
from tests.test_delegated_run_discovery import CONT, NANNY, ROOT, _drive, _execute, _receipts, _top

geo = discovery.geo  # pytest fixture, also restores coverage globals


def _old_receipt(geo, key, status="completed"):
    text = json.dumps({"primary_output": "🌒漢é" * 17000 + "exact tail"}, ensure_ascii=False)
    artifact = retain_terminal_bytes(geo.canonical, CONT, "retained", text, text.encode())
    ref = {**artifact["source"], "complete": True,
           "read": {"tool": "read_file", "arguments": {
               "root": "artifact_store", "path": artifact["path"], "start_line": 1, "max_lines": 2000}}}
    if "observation" in key:
        ref["content_kind"] = "terminal_observation"
    row = {"run_id": "retained", "state": "succeeded", key: [ref] if key.startswith("other_") else ref}
    write_task_result(geo.canonical, NANNY, status, result="nanny ended", parent_task_id=ROOT,
                      root_task_id=ROOT, delegate_terminal_reconciliation={"audit_status": "ok", "terminal_runs": [row]})
    _drive(geo, NANNY)
    shutil.rmtree(geo.tmp / "state" / "headless_tasks" / NANNY)
    assert (task_artifact_dir_path(geo.canonical, CONT) / ref["path"]).exists()
    return text, ref


def _whole_from_first(reader, ref):
    first = ref["read"]
    assert first == ref["source_read"] and first["tool"] == "get_task_result"
    assert first["arguments"]["task_id"] == NANNY != ref["task_id"] == CONT
    response = _execute(reader, first["tool"], **first["arguments"])
    assert response.status == "ok", response.text
    metadata = json.loads(response.text)["delegated_output_source"]
    assert metadata["reason"] == "source_range_required" and "text" not in metadata
    pieces = []
    for start in range(0, metadata["complete_chars"], 19001):
        result = _execute(reader, first["tool"], **first["arguments"], source_start_char=start,
                          source_end_char=min(start + 19001, metadata["complete_chars"]))
        assert result.status == "ok", result.text
        page = json.loads(result.text)["delegated_output_source"]
        assert page["complete_sha256"] == metadata["complete_sha256"]
        pieces.append(page["text"])
    return "".join(pieces)


@pytest.mark.parametrize("key", ["staged_output", "other_staged_outputs", "terminal_observation", "other_terminal_observations"])
@pytest.mark.parametrize("geometry", ["parent", "forked_parent", "root_continue"])
def test_legacy_receipt_first_recipe_opens_after_gc_with_different_staging_and_publisher(geo, key, geometry):
    text, _ref = _old_receipt(geo, key)
    old = load_task_result(geo.canonical, NANNY)
    reader = (_top(geo, CONT, forked=True, predecessor=ROOT) if geometry == "root_continue"
              else _top(geo, ROOT, forked=geometry == "forked_parent"))
    for view in ("summary", "evidence", "full"):
        response = _execute(reader, "get_task_result", task_id=NANNY, view=view)
        if view != "full":
            summary = json.loads(response.text)
        else:
            summary = json.loads(response.text.split("[SUBTASK_OUTCOME]\n")[1].split("\n[/SUBTASK_OUTCOME]")[0])
        run = summary["delegated_custody"]["terminal_runs"][0]
        ref = run[key][0] if key.startswith("other_") else run[key]
        assert _whole_from_first(reader, ref) == text
        assert ref["direct_read"]["tool"] == "read_file"
    # The authority read applies the same READ projection, not a stored migration.
    authority = json.loads(_execute(reader, "get_task_result", task_id=NANNY, include_authority=True).text)["authority"]
    run = authority["delegate_terminal_reconciliation"]["terminal_runs"][0]
    assert _whole_from_first(reader, run[key][0] if key.startswith("other_") else run[key]) == text
    assert load_task_result(geo.canonical, NANNY) == old
    assert _receipts(geo) == [], "portable discovery never earns D7"


def test_predecessor_first_recipe_keeps_physical_publisher_not_effective_successor(geo):
    text, _ = _old_receipt(geo, "staged_output", status="interrupted")
    write_task_result(geo.canonical, NANNY, "interrupted", retry_task_id="successor", superseded_by="successor")
    write_task_result(geo.canonical, "successor", "completed", result="new answer",
                      original_task_id=NANNY, timeout_retry_from=NANNY, supersedes_task_id=NANNY)
    reader = _top(geo, ROOT, forked=True)
    value = json.loads(_execute(reader, "get_task_result", task_id=NANNY, view="summary").text)
    ref = value["delegated_custody"]["predecessor_terminal_runs"][0]["staged_output"]
    assert _whole_from_first(reader, ref) == text


def test_wrap_up_prompt_puts_portable_recipe_before_large_receipt_facts(geo):
    text, ref = _old_receipt(geo, "staged_output")
    row = {"run_id": "retained", "state": "succeeded", "owner_task_id": NANNY,
           "staged_output": ref, "engine_locator": {"padding": "long" * 5000}}
    prompt = _prompt_run(row)
    prefix = "Read retained source (metadata, then source_start_char/source_end_char): "
    assert prompt.startswith(prefix)
    recipe = json.loads(prompt.removeprefix(prefix).split("\n", 1)[0])
    assert _whole_from_first(_top(geo, ROOT, forked=True), {**ref, "read": recipe, "source_read": recipe}) == text
    assert "receipt characters omitted" in prompt


def test_wrap_up_own_first_reader_uses_actual_receipt_task_and_earns_eof(geo, monkeypatch):
    from ouroboros.owner_stop_receipt import receipt_text

    reader = _top(geo, ROOT, forked=True)
    discovery._stage(geo, reader, monkeypatch)
    audit = discovery._publish_terminal_result(geo, ROOT)
    own = {**audit["terminal_runs"][0], "owner_task_id": ROOT}
    audit["owner_stop_children"] = {"task_id": ROOT, "runs": [own], "children": []}
    prompt = receipt_text(audit, prompt=True)
    prefix = "Read retained source to EOF with read_file: "
    assert prefix in prompt
    recipe = json.loads(prompt.split(prefix, 1)[1].split("\n", 1)[0])
    assert recipe["tool"] == "read_file"
    discovery._read_whole(reader, recipe["arguments"])
    assert [row["reader_task_id"] for row in _receipts(geo)] == [ROOT]


def test_wrap_up_predecessor_keeps_portable_first_even_inside_root_runs(geo):
    from ouroboros.owner_stop_receipt import receipt_text

    text, ref = _old_receipt(geo, "staged_output")
    row = {"run_id": "retained", "owner_task_id": NANNY, "staged_output": ref}
    audit = {"owner_stop_children": {"task_id": CONT, "runs": [row], "children": []}}
    prompt = receipt_text(audit, prompt=True)
    prefix = "Read retained source (metadata, then source_start_char/source_end_char): "
    recipe = json.loads(prompt.split(prefix, 1)[1].split("\n", 1)[0])
    assert _whole_from_first(_top(geo, CONT, forked=True, predecessor=ROOT),
                             {**ref, "read": recipe, "source_read": recipe}) == text
    assert _receipts(geo) == []


def test_wrap_up_child_first_reader_uses_child_publisher_without_an_owner_field(geo):
    from ouroboros.owner_stop_receipt import receipt_text

    text, ref = _old_receipt(geo, "staged_output")
    row = {"run_id": "retained", "staged_output": ref}
    audit = {"owner_stop_children": {"task_id": ROOT, "runs": [], "children": [
        {"task_id": NANNY, "custody": {"terminal_runs": [row]}}]}}
    prompt = receipt_text(audit, prompt=True)
    prefix = "Read retained source (metadata, then source_start_char/source_end_char): "
    recipe = json.loads(prompt.split(prefix, 1)[1].split("\n", 1)[0])
    assert _whole_from_first(_top(geo, ROOT, forked=True),
                             {**ref, "read": recipe, "source_read": recipe}) == text
    assert _receipts(geo) == []
