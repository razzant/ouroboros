"""Exact retained-call reads through the task result's selected authority."""
from pathlib import Path
import json
import pytest
from ouroboros.headless import copy_child_task_result, prepare_task_drive, prune_headless_task_drives, retry_child_task_refs
from ouroboros.observability import persist_call
from ouroboros.task_results import STATUS_COMPLETED, write_task_result
from ouroboros.tools.control_task_results import _get_task_result
from ouroboros.tools.registry import ToolContext
from ouroboros.tools.tool_result import _published_tool_result


def _ctx(root, task_id="parent", **metadata):
    repo = root / "repo"
    repo.mkdir(exist_ok=True)
    return ToolContext(repo_dir=repo, drive_root=root, task_id=task_id,
                       budget_drive_root=str(root), task_metadata={"budget_drive_root": str(root), **metadata})


def _call(ctx, **arguments):
    ctx._active_builtin_tool_result = None
    text = _get_task_result(ctx, **arguments)
    result = _published_tool_result(ctx, None)
    try:
        source = json.loads(text)["call_source"]
    except (ValueError, KeyError):
        source = {}
    return {"status": result.status, "code": result.code, "message": text}, {"source": source}


def test_digest_addressed_source_uses_redacted_projection_after_promotion_pack_and_cleanup(tmp_path):
    from ouroboros import evidence_packs as packs

    parent = tmp_path / "data"
    parent.mkdir()
    task_id = "retained-child"
    child = prepare_task_drive(parent, task_id, "empty")
    assert child is not None
    trace = persist_call(
        child, task_id=task_id, call_id="tool_exact", call_type="tool_call",
        payload={"tool": "demo", "token": "RAW-SECRET-MUST-NOT-LEAK", "result": "published"},
        keep_raw=True,
    )
    write_task_result(child, task_id, STATUS_COMPLETED, result="published", terminal_origin="model_final",
                      trace_refs={"tool_call_refs": [{"manifest_ref": trace["manifest_ref"],
                                                       "redacted_projection_ref": trace["redacted_projection_ref"]}]})
    copied = copy_child_task_result(parent, {"id": task_id, "drive_root": str(child)})
    copied = retry_child_task_refs(parent, child, task_id)
    assert copied and copied["child_ref_promotion"]["status"] == "complete"
    manifest_ref = copied["trace_refs"]["tool_call_refs"][0]["manifest_ref"]

    members = packs.loose_members(parent, task_id)
    assert members
    pack_path, sidecar = packs.write_pack(parent, task_id, 1, members)
    packs.index_upsert(parent, pack_path, sidecar)
    for member in members:
        source = Path(member["source"])
        if source.is_file() and source.is_relative_to(parent / "observability/blobs"):
            source.unlink()
    report = prune_headless_task_drives(parent, retention_days=7, now=4_000_000_000.0, live=lambda _task: False)
    assert any(row["task_id"] == task_id for row in report["pruned"])
    assert not child.exists()

    ctx = _ctx(parent)
    meta_result, meta = _call(
        ctx, task_id=task_id, call_id="tool_exact",
        manifest_sha256=manifest_ref["sha256"],
    )
    assert meta_result["status"] == "ok" and "text" not in meta["source"]
    source_result, source = _call(
        ctx, task_id=task_id, call_id="tool_exact",
        manifest_sha256=manifest_ref["sha256"], source_start_char=0,
        source_end_char=meta["source"]["complete_chars"],
    )
    assert source_result["status"] == "ok"
    assert source["source"]["raw_payload_used"] is False
    assert "RAW-SECRET-MUST-NOT-LEAK" not in source["source"]["text"]
    assert "***REDACTED[" in source["source"]["text"]


def test_dispatch_source_stays_on_canonical_plane_when_execution_result_is_selected(tmp_path):
    from ouroboros.tool_call_log import new_invocation, persist_dispatch_source

    task_id = "dispatch-plane"
    execution = prepare_task_drive(tmp_path, task_id, "empty")
    assert execution is not None
    invocation = new_invocation("provider-call", {}, 1)
    persist_dispatch_source(
        {"budget_drive_root": str(tmp_path)}, execution / "logs", task_id,
        invocation, "run_script", {"script": "print('canonical-dispatch')"},
    )
    write_task_result(tmp_path, task_id, "running", result="not final")
    write_task_result(execution, task_id, STATUS_COMPLETED, result="execution final")

    ref = invocation["args_source_ref"]
    result, metadata = _call(
        _ctx(tmp_path), task_id=task_id,
        call_id=ref["call_id"], manifest_sha256=ref["sha256"],
    )
    assert result["status"] == "ok"
    assert metadata["source"]["call_type"] == "tool_dispatch"


def test_selected_retry_call_reads_through_the_original_id_and_nothing_else(tmp_path):
    retry = persist_call(tmp_path, task_id="retry", call_id="tool_retry", call_type="tool_call",
                         payload={"tool": "read_file", "result": "retry bytes"})
    old = persist_call(tmp_path, task_id="original", call_id="tool_old", call_type="tool_call",
                       payload={"tool": "read_file", "result": "superseded bytes"})
    write_task_result(tmp_path, "retry", STATUS_COMPLETED, result="retry answer", terminal_origin="model_final",
                      trace_refs={"tool_call_refs": [{"manifest_ref": retry["manifest_ref"]}]})
    write_task_result(tmp_path, "original", "failed", result="old answer", retry_task_id="retry",
                      trace_refs={"tool_call_refs": [{"manifest_ref": old["manifest_ref"]}]})
    ctx = _ctx(tmp_path)
    digest = retry["manifest_ref"]["sha256"]
    result, metadata = _call(ctx, task_id="original", call_id="tool_retry", manifest_sha256=digest)
    assert result["status"] == "ok", result
    assert metadata["source"]["task_id"] == "retry"
    result, source = _call(ctx, task_id="original", call_id="tool_retry", manifest_sha256=digest,
                           source_start_char=0, source_end_char=metadata["source"]["complete_chars"])
    assert result["status"] == "ok" and "retry bytes" in source["source"]["text"]
    # The superseded attempt's own call is not a second selection behind the retry's answer.
    refused, _ = _call(ctx, task_id="original", call_id="tool_old", manifest_sha256=old["manifest_ref"]["sha256"])
    assert refused["status"] == "blocked" and refused["code"] == "ACCESS_BLOCKED"


def test_read_call_verifies_decoded_task_call_and_family_before_serving_projection(tmp_path):
    task_id = "manifest-identity"
    expected = persist_call(
        tmp_path, task_id=task_id, call_id="expected_call", call_type="tool_call",
        payload={"result": "EXPECTED"},
    )
    foreign_call = persist_call(
        tmp_path, task_id=task_id, call_id="other_call", call_type="tool_call",
        payload={"result": "OTHER-CALL-MUST-NOT-BE-SERVED"},
    )
    wrong_family = persist_call(
        tmp_path, task_id=task_id, call_id="response_in_tool_slot", call_type="llm_response",
        payload={"message": "LLM-RESPONSE-MUST-NOT-BE-SERVED-AS-TOOL"},
    )
    swapped_ref = {
        **foreign_call["manifest_ref"],
        # The published address claims the requested call while its physical
        # manifest and digest belong to another call.
        "call_id": expected["manifest_ref"]["call_id"],
    }
    write_task_result(tmp_path, task_id, STATUS_COMPLETED, result="done", trace_refs={
        "tool_call_refs": [
            {"manifest_ref": swapped_ref},
            {"manifest_ref": wrong_family["manifest_ref"]},
        ],
    })
    ctx = _ctx(tmp_path)

    identity_failed, _data = _call(
        ctx, task_id=task_id, call_id="expected_call",
        manifest_sha256=foreign_call["manifest_ref"]["sha256"],
    )
    family_failed, _data = _call(
        ctx, task_id=task_id, call_id="response_in_tool_slot",
        manifest_sha256=wrong_family["manifest_ref"]["sha256"],
    )
    for failed, forbidden in (
        (identity_failed, "OTHER-CALL-MUST-NOT-BE-SERVED"),
        (family_failed, "LLM-RESPONSE-MUST-NOT-BE-SERVED-AS-TOOL"),
    ):
        assert failed["status"] == "error" and failed["code"] == "TOOL_ERROR"
        assert forbidden not in json.dumps(failed)


def test_existing_manifest_reader_adds_optional_physical_admission_and_size_without_changing_default(tmp_path):
    from ouroboros.observability import read_call_manifest_ref

    retained = persist_call(
        tmp_path, task_id="reader-options", call_id="bounded_manifest", call_type="tool_call",
        payload={"result": "visible"},
    )
    ref = retained["manifest_ref"]
    assert read_call_manifest_ref(tmp_path, ref, task_id="reader-options")["call_id"] == "bounded_manifest"
    admitted = []
    bounded = read_call_manifest_ref(
        tmp_path, ref, task_id="reader-options", max_bytes=16 * 1024,
        admit=lambda path: admitted.append(Path(path)), call_types=("tool_call",),
    )
    assert bounded["call_type"] == "tool_call" and admitted == [Path(ref["path"])]
    with pytest.raises(PermissionError, match="controlled admission refusal"):
        read_call_manifest_ref(
            tmp_path, ref, task_id="reader-options", max_bytes=16 * 1024,
            admit=lambda _path: (_ for _ in ()).throw(PermissionError("controlled admission refusal")),
        )
    with pytest.raises(ValueError, match="byte limit"):
        read_call_manifest_ref(tmp_path, ref, task_id="reader-options", max_bytes=8)


def test_source_range_errors_are_typed_and_raw_only_legacy_manifest_is_unavailable(tmp_path):
    from ouroboros.observability import write_blob, write_call_manifest

    task_id = "source-errors"
    visible = persist_call(tmp_path, task_id=task_id, call_id="visible", call_type="tool_call",
                           payload={"result": "bounded"})
    corrupt = persist_call(tmp_path, task_id=task_id, call_id="corrupt", call_type="tool_call",
                           payload={"result": "must fail verification"})
    raw_ref = write_blob(tmp_path, {"token": "RAW-ONLY"})
    raw_manifest = write_call_manifest(
        tmp_path, task_id=task_id, call_id="raw_only",
        manifest={"call_type": "tool_call", "full_payload_ref": raw_ref, "full_payload_redacted": False},
    )
    write_task_result(tmp_path, task_id, STATUS_COMPLETED, result="done", trace_refs={
        "tool_call_refs": [
            {"manifest_ref": visible["manifest_ref"]},
            {"manifest_ref": corrupt["manifest_ref"]},
            {"manifest_ref": raw_manifest},
        ],
    })
    ctx = _ctx(tmp_path)
    invalid, _data = _call(
        ctx, task_id=task_id, call_id="visible",
        manifest_sha256=visible["manifest_ref"]["sha256"], source_start_char=0, source_end_char=10_000,
    )
    assert invalid["status"] == "error" and invalid["code"] == "TOOL_ARG_ERROR"
    assert "complete_chars" in invalid["message"] and "text" not in _data["source"]

    unavailable, _data = _call(
        ctx, task_id=task_id, call_id="raw_only",
        manifest_sha256=raw_manifest["sha256"],
    )
    assert unavailable["status"] == "unavailable" and unavailable["code"] == "LEGACY_UNAVAILABLE"
    assert "no available redacted projection" in unavailable["message"]
    assert "RAW-ONLY" not in json.dumps(unavailable)

    Path(corrupt["redacted_projection_ref"]["path"]).write_bytes(b"corrupt-cas-bytes")
    failed, _data = _call(
        ctx, task_id=task_id, call_id="corrupt",
        manifest_sha256=corrupt["manifest_ref"]["sha256"],
    )
    assert failed["status"] == "error" and failed["code"] == "TOOL_ERROR"
    assert failed["message"] == "Retained call source is unreadable or failed verification."
    assert str(tmp_path) not in json.dumps(failed)


def test_manifest_permission_error_falls_back_but_admission_refusal_does_not(tmp_path, monkeypatch):
    from ouroboros.observability import read_call_manifest_ref
    from ouroboros.observability import write_blob

    retained = persist_call(tmp_path, task_id="task", call_id="call", call_type="tool_call", payload={"result": "ok"})
    ref = retained["manifest_ref"]
    write_blob(tmp_path, Path(ref["path"]).read_text(), kind="txt")
    read = Path.read_bytes
    def denied(path):
        if path == Path(ref["path"]):
            raise PermissionError("filesystem denial")
        return read(path)
    monkeypatch.setattr(Path, "read_bytes", denied)
    assert read_call_manifest_ref(tmp_path, ref, task_id="task")["call_id"] == "call"
    admitted = []
    assert read_call_manifest_ref(tmp_path, ref, task_id="task", admit=admitted.append)["call_id"] == "call"
    assert len(admitted) >= 2
    def refuse(path):
        raise PermissionError("admission denied")
    with pytest.raises(PermissionError, match="admission denied"):
        read_call_manifest_ref(tmp_path, ref, task_id="task", admit=refuse)


def test_retained_source_obeys_resource_and_presence_scope(tmp_path, monkeypatch):
    import ouroboros.presence_authority as presence
    retained = persist_call(tmp_path, task_id="task", call_id="call", call_type="tool_call", payload={"result": "ok"})
    ref = retained["manifest_ref"]
    write_task_result(tmp_path, "task", STATUS_COMPLETED, result="done",
                      trace_refs={"tool_call_refs": [{"manifest_ref": ref}]})
    ctx = _ctx(tmp_path)
    args = {"task_id": "task", "call_id": "call", "manifest_sha256": ref["sha256"]}
    assert _call(ctx, **args)[0]["status"] == "ok"
    for key in ("manifest_ref", "redacted_projection_ref"):
        ctx.task_contract = {"resource_policy": {"protected_artifacts": [{
            "paths": [retained[key]["path"]], "role": "black_box_reference"}]}}
        assert _call(ctx, **args)[0]["status"] == "blocked"
    ctx.task_contract = {}
    assert _call(ctx, **args, presence_scope="own_binding")[0]["status"] == "blocked"
    monkeypatch.setattr(presence, "presence_caller_binding", lambda _ctx: "binding")
    monkeypatch.setattr(presence, "presence_work_refusal", lambda *_a, **_kw: "")
    monkeypatch.setattr(presence, "presence_effective_refusal", lambda *_a, **_kw: "")
    assert _call(ctx, **args, presence_scope="own_binding")[0]["status"] == "ok"
    monkeypatch.setattr(presence, "presence_work_refusal", lambda *_a, **_kw: "outside binding")
    assert _call(ctx, **args, presence_scope="own_binding")[0]["status"] == "blocked"


@pytest.mark.parametrize("extra", [{"manifest_sha256": "x"}, {"include_authority": True}, {"view": "answer"}])
def test_call_arguments_refuse_before_selection(tmp_path, monkeypatch, extra):
    import ouroboros.tools.control_task_results as results
    monkeypatch.setattr(results, "load_effective_task_result", lambda *_a: pytest.fail("selected before validation"))
    result, _ = _call(_ctx(tmp_path), **{"task_id": "task", "call_id": "call", "manifest_sha256": "a" * 64, **extra})
    assert result["code"] == "TOOL_ARG_ERROR"
