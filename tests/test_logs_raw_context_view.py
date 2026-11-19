"""Raw for a context view checkpoint: the stored final event, plus what its own ``at_source`` left behind.

The producer (``loop_round_limits._run_authored_context_view``) writes the final event with the structures
that equal its saved ``candidate_materialization`` receipt moved behind ``receipt_source``. Raw re-reads
that stored event by its exact bytes and hydrates only those declared, absent fragments. A reference names
the stored row and nothing else; a receipt Raw cannot use is an explicit gap over the stored event, served whole.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from ouroboros.outcomes import public_task_result
from tests._logs_raw_shared import client, preview, read, write
from tests._store_shared import seed_result


def _messages():
    return [{"role": "system", "content": "Full unchanged identity/books"},
            {"role": "user", "content": "Owner task"},
            {"role": "assistant", "tool_calls": [{"id": "read", "function": {"name": "read_file", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "read", "content": "Long exact evidence " * 200}]


def produce(drive, task_id):
    """The real producer and the real emitter: the event lands in ``drive/logs/events.jsonl``."""
    from ouroboros.loop_round_limits import _CompactionRoundContext, _run_authored_context_view
    from ouroboros.tools.compact_context import _compact_context, record_context_view

    ctx = SimpleNamespace(active_context_mode="low")
    messages, schemas = _messages(), []
    record_context_view(ctx, messages, schemas)
    _compact_context(ctx, working_note="I retain the conclusion.")
    registry = SimpleNamespace(_ctx=ctx, schemas=lambda: schemas)
    frame = _CompactionRoundContext(registry, drive, drive / "logs", task_id, 1, None, lambda text: None, schemas,
                                    lambda candidate, tools: {"accepted": True})
    _run_authored_context_view(messages, frame, ctx._pending_compaction, None)
    stored = [json.loads(line) for line in (drive / "logs" / "events.jsonl").read_text().splitlines()]
    event, = [row for row in stored if row.get("checkpoint_kind") == "context_view"]
    assert event["at_source"] and event["receipt_source"]["task_id"] == task_id, "the producer moved structures"
    return event


def receipt_of(root, event):
    from ouroboros.artifacts import read_actor_source_bytes

    return json.loads(read_actor_source_bytes(root, event["receipt_source"]["task_id"], event["receipt_source"]))


def expected(event, saved):
    whole = dict(event)
    for key in event["at_source"]:
        whole.setdefault(key, saved["receipt"][key])
    return public_task_result(whole, include_outcome_axes=True)


def pages(http, row):
    text, offset = "", 0
    while True:
        response = read(http, row, offset=offset, limit=4096)
        assert response.status_code == 200, response.text
        page = response.json()
        assert page["source"] == row["_raw"]["source"] and page["representation"] == row["_raw"]["representation"]
        assert (page["unavailable_fields"], page["unavailable_reason"]) == ([], "")
        text += page["text"]
        if not page["has_more"]:
            return text, page
        offset = page["next_offset"]


def context_row(rows):
    found, = [row for row in rows if row.get("checkpoint_kind") == "context_view"]
    return found


@pytest.mark.parametrize("scope", ["unfiltered", "own_task", "parent"])
def test_default_compact_reference_reconstructs_the_producers_final_event(tmp_path, scope):
    task_id = "view-task"
    drive, query = tmp_path, {}
    if scope == "parent":
        drive = tmp_path / "task_drives" / task_id
        seed_result(tmp_path, "parent", {"status": "running"})
        admission = {"parent_task_id": "parent", "root_task_id": "parent", "delegation_role": "subagent",
                     "child_drive_root": str(drive), "status": "running"}
        seed_result(tmp_path, task_id, admission)
        seed_result(drive, task_id, {**admission, "task_id": task_id})
        query = {"task_id": "parent"}
    elif scope == "own_task":
        seed_result(tmp_path, task_id, {"status": "running"})
        query = {"task_id": task_id}
    event = produce(drive, task_id)
    saved = receipt_of(tmp_path, event)
    http = client(tmp_path)
    row = context_row(preview(http, **query))
    raw = row["_raw"]
    assert raw["source"]["kind"] == "log_row" and raw["representation"] == "reconstructed"
    assert raw["source"]["task_id"] == query.get("task_id", ""), "the reference keeps the query's own scope"
    assert set(raw["source"]) == {"v", "kind", "placement", "task_id", "log", "start", "length", "sha256"}
    assert raw["public_bytes"] is None, "the receipt is read by Raw, not by the preview"
    assert set(event["at_source"]) <= set(raw["omitted_fields"]), "the preview names what only Raw holds"
    text, last = pages(http, row)
    reconstructed = json.loads(text)
    assert reconstructed == json.loads(json.dumps(expected(event, saved)))
    for key, value in public_task_result(event, include_outcome_axes=True).items():
        assert reconstructed[key] == value, f"final event field {key} is served as stored"
    for key, count in event["at_source"].items():
        assert len(reconstructed[key]) == count
    whole = read(http, row, download=1)
    assert whole.status_code == 200 and whole.content == text.encode()
    assert whole.headers["x-ouroboros-raw-representation"] == "reconstructed"
    assert whole.headers["x-ouroboros-source-sha256"] == raw["source"]["sha256"]
    assert whole.headers["x-ouroboros-payload-sha256"] == last["sha256"] == hashlib.sha256(whole.content).hexdigest()


def test_the_producers_own_retention_failure_is_one_whole_plain_row(tmp_path):
    """The producer cannot write its receipt (a read-only checkpoint directory, a real OSError): its final event
    keeps every structure inline, declares nothing, and Raw serves that row's own public projection, whole."""
    import os
    import shutil

    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.loop_round_limits import _CompactionRoundContext, _run_authored_context_view
    from ouroboros.tools.compact_context import _compact_context, record_context_view

    task_id = "fail-task"

    def view(drive):
        ctx = SimpleNamespace(active_context_mode="low")
        messages = _messages()
        record_context_view(ctx, messages, [])
        _compact_context(ctx, working_note="I retain the conclusion.", keep_unit_ids=[])
        frame = _CompactionRoundContext(SimpleNamespace(_ctx=ctx, schemas=lambda: []), drive, drive / "logs", task_id, 1,
                                        None, lambda text: None, [], lambda candidate, tools: {"accepted": True})
        candidate = _run_authored_context_view(messages, frame, ctx._pending_compaction, None)
        rows = [json.loads(line) for line in (drive / "logs" / "events.jsonl").read_text().splitlines()]
        event, = [row for row in rows if row.get("checkpoint_kind") == "context_view"]
        return event, candidate[-1]["content"]

    # The same view in a separate root names its compaction checkpoint; the store is write-once, so the
    # served root holding those exact bytes is not rewritten, and only the receipt write meets the refusal.
    dry, _ = view(tmp_path / "dry")
    checkpoint = json.loads((task_artifact_dir_path(tmp_path / "dry", task_id) / dry["receipt_source"]["path"])
                            .read_bytes())["receipt"]["checkpoint_ref"]
    drive = tmp_path / "served"
    folder = task_artifact_dir_path(drive, task_id, create=True) / "source_handles" / "context_checkpoints"
    folder.mkdir(parents=True)
    shutil.copyfile(task_artifact_dir_path(tmp_path / "dry", task_id) / checkpoint["path"],
                    task_artifact_dir_path(drive, task_id) / checkpoint["path"])
    os.chmod(folder, 0o500)
    try:
        try:
            (folder / ".probe").write_bytes(b"")
        except OSError:
            pass
        else:
            pytest.skip("this platform or user writes into a 0500 directory")
        event, notice = view(drive)
    finally:
        os.chmod(folder, 0o700)
    assert event["status"] == "applied" and event["checkpoint_ref"]["sha256"] == checkpoint["sha256"]
    assert '"receipt_source": "unavailable"' in notice, "the model was told the receipt was not retained"
    assert "receipt_source" not in event and "at_source" not in event, "nothing is declared behind a receipt"
    for key in dry["at_source"]:
        assert len(event[key]) == dry["at_source"][key], f"{key} stays inline, whole"
    http = client(drive)
    row = context_row(preview(http))
    raw = row["_raw"]
    assert raw["representation"] in {"stored", "public_task_result"} and raw["public_bytes"] is not None
    whole = read(http, row, download=1)
    assert whole.status_code == 200 and "x-ouroboros-raw-unavailable" not in whole.headers
    assert len(whole.content) == raw["public_bytes"]
    assert json.loads(whole.content) == stored_projection(event)
    if raw["representation"] == "stored":
        log = (drive / "logs" / "events.jsonl").read_bytes()
        assert whole.content == log[raw["source"]["start"]:raw["source"]["start"] + raw["source"]["length"]]


def stored_event(root, task_id="view-task", *, receipt=None, stage="candidate_materialization", published=False,
                 receipt_task=None, data=None, **fields):
    """A context view event row written as the producer writes it, with its receipt stored for real."""
    from ouroboros.artifacts import store_actor_source_bytes

    receipt = receipt if receipt is not None else {"status": "applied", "source_refs": [{"unit": "u1"}, {"unit": "u2"}],
                                                   "selected_unit_ids": ["u1"]}
    if data is None:
        data = json.dumps({"stage": stage, "published": published, "receipt": receipt}, sort_keys=True).encode()
    source = store_actor_source_bytes(root, receipt_task or task_id, category="context_checkpoints",
                                      source_id=hashlib.sha256(data).hexdigest(), data=data, extension="json")
    receipt_source = {**source, "task_id": receipt_task or task_id}
    receipt_source["read"] = {"tool": "memory_read", "arguments": {"source_ref": dict(receipt_source)}}
    event = {"ts": "2026-11-06T00:00:00Z", "type": "task_checkpoint", "task_id": task_id,
             "checkpoint_kind": "context_view", "round": 1, "status": "applied",
             "at_source": {"source_refs": 2, "selected_unit_ids": 1}, "receipt_source": receipt_source,
             "receipt_stage": "candidate_materialization", **fields}
    write(root, event, append=(root / "logs" / "events.jsonl").exists())
    return event, source


def rewritten_ref(row, source):
    encoded = json.dumps(source, separators=(",", ":")).encode()
    return {"_raw": {"ref": base64.urlsafe_b64encode(encoded).decode().rstrip("="), "source": source}}


def test_a_final_event_key_is_never_overwritten_by_its_receipt(tmp_path):
    event, _ = stored_event(tmp_path, selected_unit_ids=["final-choice"])
    row = context_row(preview(client(tmp_path)))
    body = json.loads(read(client(tmp_path), row, download=1).content)
    assert body["selected_unit_ids"] == ["final-choice"], "the event's final value wins over the saved candidate"
    assert body["source_refs"] == [{"unit": "u1"}, {"unit": "u2"}], "an absent declared fragment is hydrated"
    assert body["at_source"] == event["at_source"] and body["receipt_source"] == event["receipt_source"]


def test_the_public_projection_applies_to_hydrated_fragments_too(tmp_path):
    receipt = {"status": "applied", "source_refs": [{"unit": "u1", "result_status": "private"}, {"unit": "u2"}],
               "selected_unit_ids": ["u1"]}
    stored_event(tmp_path, receipt=receipt)
    body = json.loads(read(client(tmp_path), context_row(preview(client(tmp_path))), download=1).content)
    assert body["source_refs"] == [{"unit": "u1"}, {"unit": "u2"}], "Raw's public projection covers the whole row"


def test_an_event_without_outcome_fields_still_previews(tmp_path):
    event, _ = stored_event(tmp_path)
    del event["status"]
    (tmp_path / "logs" / "events.jsonl").write_text(json.dumps(event) + "\n")
    http = client(tmp_path)
    row = context_row(preview(http))
    assert row["_raw"]["representation"] == "reconstructed"
    assert read(http, row).status_code == 200


def test_a_reference_names_only_its_stored_row_and_carries_no_declarations(tmp_path):
    event, _ = stored_event(tmp_path)
    write(tmp_path, {"ts": "2026-11-06T00:00:01Z", "task_id": "view-task", "type": "plain", "value": 1}, append=True)
    http = client(tmp_path)
    rows = preview(http)
    row, plain = context_row(rows), next(item for item in rows if item.get("type") == "plain")
    assert plain["_raw"]["source"]["kind"] == row["_raw"]["source"]["kind"] == "log_row"
    assert plain["_raw"]["representation"] == "stored" and row["_raw"]["representation"] == "reconstructed"
    plain_body = read(http, plain, download=1)
    assert plain_body.headers["x-ouroboros-raw-representation"] == "stored" and json.loads(plain_body.content)["value"] == 1
    # One identity: no second kind chooses a representation, and declarations are the stored row's own.
    for target in (row, plain):
        assert read(http, rewritten_ref(target, {**target["_raw"]["source"], "kind": "context_view"})).status_code == 400
    for extra in ({"at_source": {"source_refs": 1}}, {"receipt_source": event["receipt_source"]},
                  {"event": dict(row["_raw"]["source"])}):
        forged = rewritten_ref(row, {**row["_raw"]["source"], **extra})
        assert read(http, forged).status_code == 400, extra


def stored_projection(event):
    return json.loads(json.dumps(public_task_result(event, include_outcome_axes=True)))


@pytest.mark.parametrize("damage", ["count", "stage", "published", "not_structure", "receipt_bytes",
                                    "list", "not_json", "string", "empty", "receipt_list"])
def test_a_receipt_that_does_not_match_leaves_the_stored_event_readable_with_an_explicit_gap(tmp_path, damage):
    receipt = {"status": "applied", "source_refs": [{"unit": "u1"}, {"unit": "u2"}], "selected_unit_ids": ["u1"]}
    options = {}
    if damage == "count":
        receipt["source_refs"] = [{"unit": "u1"}]
    elif damage == "stage":
        options["stage"] = "published_view"
    elif damage == "published":
        options["published"] = True
    elif damage == "not_structure":
        receipt["selected_unit_ids"] = "u"
    else:
        options["data"] = {"list": b"[1, 2]", "not_json": b"not json", "string": b'"candidate_materialization"',
                           "empty": b"{}", "receipt_list": b'{"stage": "candidate_materialization", "published": false, '
                                                           b'"receipt": []}'}.get(damage)
    event, source = stored_event(tmp_path, receipt=receipt, **options)
    http = client(tmp_path)
    row = context_row(preview(http))
    if damage == "receipt_bytes":
        from ouroboros.artifacts import task_artifact_dir_path

        target = task_artifact_dir_path(tmp_path, "view-task", create=False) / source["path"]
        target.write_bytes(target.read_bytes().replace(b'"u2"', b'"u3"'))
    response = read(http, row)
    assert response.status_code == 200, response.text
    page = response.json()
    assert page["representation"] == "reconstructed" and page["unavailable_reason"] == "receipt_mismatch"
    assert page["unavailable_fields"] == ["source_refs", "selected_unit_ids"], "every declared field it lacks, named"
    assert json.loads(page["text"]) == stored_projection(event), "the stored event, whole, and nothing else"
    assert response.headers["x-ouroboros-raw-unavailable"].startswith("receipt_mismatch")


def test_a_missing_receipt_is_an_explicit_gap_over_the_still_readable_stored_event(tmp_path):
    from ouroboros.artifacts import task_artifact_dir_path

    event, source = stored_event(tmp_path)
    http = client(tmp_path)
    row = context_row(preview(http))
    path = task_artifact_dir_path(tmp_path, "view-task", create=False) / source["path"]
    kept = path.read_bytes()
    path.unlink()
    response = read(http, row)
    assert response.status_code == 200, response.text
    page = response.json()
    assert (page["representation"], page["unavailable_reason"]) == ("reconstructed", "receipt_unavailable")
    assert page["unavailable_fields"] == ["source_refs", "selected_unit_ids"]
    assert json.loads(page["text"]) == stored_projection(event)
    whole = read(http, row, download=1)
    assert whole.content == page["text"].encode() and whole.headers["x-ouroboros-raw-unavailable"].startswith("receipt_unavailable")
    # The receipt back: the complete reconstruction is a different body, so a reader pinned to the first is told so.
    path.write_bytes(kept)
    complete = read(http, row).json()
    assert complete["unavailable_fields"] == [] and complete["sha256"] != page["sha256"]
    assert complete["source"] == page["source"]


def test_a_reconstruction_with_nothing_absent_reads_no_receipt(tmp_path):
    from ouroboros.artifacts import task_artifact_dir_path

    _, source = stored_event(tmp_path, source_refs=["final"], selected_unit_ids=["final"])
    (task_artifact_dir_path(tmp_path, "view-task", create=False) / source["path"]).unlink()
    http = client(tmp_path)
    page = read(http, context_row(preview(http))).json()
    assert (page["representation"], page["unavailable_fields"], page["unavailable_reason"]) == ("reconstructed", [], "")
    assert json.loads(page["text"])["source_refs"] == ["final"]


def test_out_of_range_and_cross_row_references_are_typed_refusals(tmp_path):
    """A well-formed reference past the chain or inside another row never reaches a segment lookup."""
    stored_event(tmp_path)
    http = client(tmp_path)
    row = context_row(preview(http))
    source = row["_raw"]["source"]
    size = (tmp_path / "logs" / "events.jsonl").stat().st_size
    for start, length, status in ((size, 1, 404), (size + 4096, 7, 404), (0, size + 1, 404), (1, 5, 409)):
        forged = rewritten_ref(row, {**source, "start": start, "length": length})
        response = read(http, forged)
        assert response.status_code == status, (start, length, response.text)
        assert response.json()["reason_code"] == ("log_source_unavailable" if status == 404 else "log_source_changed")
    (tmp_path / "logs" / "events.jsonl").write_bytes(b"")
    response = read(http, rewritten_ref(row, source))
    assert response.status_code == 404 and response.json()["reason_code"] == "log_source_unavailable"


@pytest.mark.parametrize("declarations", [
    {"at_source": {}}, {"at_source": ["source_refs"]}, {"receipt_source": "unavailable"},
    {"checkpoint_kind": "round_limit"}, {"at_source": {"source_refs": -1}}, {"at_source": {"source_refs": True}},
    {"receipt_source": {"path": "source_handles/other/x.json"}}, {"receipt_source": {"root": "drive"}},
    {"receipt_source": {"kind": "log_row"}}, {"receipt_source": {"sha256": "not-a-digest"}},
    {"receipt_source": {"task_id": "another-task"}}, {"task_id": "another-task"},
], ids=lambda value: json.dumps(value, sort_keys=True))
def test_rows_without_valid_declarations_are_plain_rows(tmp_path, declarations):
    event, _ = stored_event(tmp_path)
    if isinstance(declarations.get("receipt_source"), dict):
        declarations = {"receipt_source": {**event["receipt_source"], **declarations["receipt_source"]}}
    (tmp_path / "logs" / "events.jsonl").write_text(json.dumps({**event, **declarations}) + "\n")
    http = client(tmp_path)
    row, = preview(http)
    assert row["_raw"]["source"]["kind"] == "log_row"
    assert row["_raw"]["representation"] in {"stored", "public_task_result"}
    assert read(http, row).status_code == 200


def test_a_changed_event_row_is_not_reconstructed(tmp_path):
    event, _ = stored_event(tmp_path)
    http = client(tmp_path)
    row = context_row(preview(http))
    changed = copy.deepcopy(event)
    changed["at_source"]["source_refs"] = 1
    (tmp_path / "logs" / "events.jsonl").write_text(json.dumps(changed) + "\n")
    response = read(http, row)
    assert response.status_code in {404, 409}, response.text


def test_published_contract_names_the_reconstruction():
    from ouroboros.gateway.logs import LogRawReference, LogRawResponse
    from ouroboros.gateway.schema import json_schema_for

    response, reference = json_schema_for(LogRawResponse), json_schema_for(LogRawReference)
    assert response["properties"]["source"]["properties"]["kind"] == {"enum": ["log_row"]}
    for schema in (response, reference):
        assert schema["properties"]["representation"] == {"enum": ["stored", "public_task_result", "reconstructed"]}
    assert {"type": "null"} in reference["properties"]["public_bytes"]["anyOf"]
    assert response["properties"]["unavailable_reason"] == {"enum": ["", "receipt_unavailable", "receipt_mismatch"]}
    assert response["properties"]["unavailable_fields"]["type"] == "array"
