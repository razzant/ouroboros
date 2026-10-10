"""Historical premises use existing consumers, with only transports scripted.

The fixtures start at the immutable source/exhibit boundary. Producer-selection
tests separately prove which author inputs enter that boundary. Native and
reflection tests inspect the NEXT model input after the real read_file result;
the offline session physically reads its advertised file. None is evidence of
a real provider, subscription harness, or model comprehension.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros.acceptance_retrieving import acceptance_retrieving_work_order
from ouroboros.artifacts import read_actor_source_bytes, store_actor_source_bytes, task_artifact_dir_path
from ouroboros.review_source_closure import retain_review_request_sources
from ouroboros.review_substrate import ReviewRequest, run_review_request
from tests.test_acceptance_delivery import _CLEAN_VERDICT, _EpisodeLLM, _fake_session, _tool_call
from tests.test_acceptance_source_first import _SOURCE_PATH, _prepared_request
from tests import test_main_authored_context as context_fixtures

main_loop = context_fixtures.main_loop


_OLD = "HISTORICAL-AUDIENCE-EXECUTIVE-74a2"
_TODAY = "CURRENT-AUDIENCE-PUBLIC-91d8"
_PREVIEW = (
    "Bounded historical author input preview. Full captured premises are not "
    "included in this packet; a source handle does not prove the evaluator read them."
)


def _historical_source(root, task_id):
    """One selected immutable source; arbitrary event fields stay opaque data."""
    foreign = store_actor_source_bytes(root, task_id, category="tool_results",
        source_id="event-lookalike", data=b"EVENT-LOOKALIKE-MUST-NOT-BE-RETAINED", extension="txt")
    payload = {
        "version": 1,
        "kind": "historical_author_input",
        "task_id": task_id,
        "selected_input": {
            "conversation": [{"role": "user", "text": "Make a presentation for " + _OLD}],
            "presence": {
                "event": {"nested": {"marker": _OLD, "source_ref": foreign}},
                "observed_text": "Please revise the feature list.",
                "instructions": "Use the role and topics captured for this turn.",
                "topics": [{"topic": "audience", "text": _OLD}],
            },
        },
        "coverage": {"selected": ["conversation", "presence"], "excluded": ["unselected_topic"]},
    }
    raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    ref = store_actor_source_bytes(root, task_id, category="context_checkpoints",
        source_id="historical-author-input", data=raw, extension="json")
    exhibit = {
        "version": 1, "status": "captured",
        "coverage": {"preview_complete": False, "evaluator_read": "not_recorded"},
        "anchors": [{"position": "first", "status": "captured", "source_ref": ref, "preview": _PREVIEW}],
    }
    return exhibit, raw, foreign


def _request(tmp_path, *, session=False, paged=False):
    request, slot, author, canonical, workspace = _prepared_request(tmp_path, session=session)
    if not paged:
        request.evidence.pop("__immutable_core_overflow__")
    exhibit, raw, foreign = _historical_source(author, request.task_id)
    request.evidence["historical_author_inputs"] = exhibit
    request.evidence["__provenance__"] = {
        **request.evidence["__provenance__"], "historical_author_inputs": "host_attested",
    }
    # Mutable same-room/profile/topic inputs deliberately disagree with history.
    (workspace / "current-author-input.json").write_text(_TODAY, encoding="utf-8")
    acceptance_retrieving_work_order(request, [slot], session_root=str(workspace), data_root=author)
    return request, slot, author, canonical, workspace, raw, foreign


def _historical_binding(request):
    anchor = request.evidence["historical_author_inputs"]["anchors"][0]
    matches = [row for row in request.policy["review_source_closure"]["refmap"]
               if row["sha256"] == anchor["source_ref"]["sha256"]]
    assert len(matches) == 1, "the historical exhibit must have one actual reader binding"
    return matches[0]


@pytest.mark.parametrize("paged", [False, True])
def test_native_historical_read_reaches_next_model_input_after_author_cleanup(tmp_path, paged):
    request, slot, author, canonical, workspace, raw, foreign = _request(tmp_path, paged=paged)

    class Reader(_EpisodeLLM):
        def _reply(self, kwargs):
            self.calls.append(copy.deepcopy(kwargs))
            if len(self.calls) == 1:
                prompt = json.dumps(kwargs["messages"], ensure_ascii=False)
                assert _OLD not in prompt and _TODAY not in prompt
                binding = _historical_binding(request)
                assert binding["retained_path"] in prompt
                assert Path(binding["retained_path"]).read_bytes() == raw
                shutil.rmtree(author)
                calls = [_tool_call("read_file", binding["read"]["arguments"], "history")]
                if paged:
                    packet_path = _SOURCE_PATH.search(kwargs["messages"][-1]["content"]).group(1)
                    calls.append(_tool_call("read_file", {"root": "artifact_store", "path": packet_path}, "packet"))
                reply = {"tool_calls": calls}
            else:
                outputs = {row["tool_call_id"]: row["content"] for row in kwargs["messages"]
                           if row.get("role") == "tool"}
                assert _OLD in outputs["history"]
                assert _TODAY not in outputs["history"]
                assert '"event"' in outputs["history"] and '"topics"' in outputs["history"]
                reply = {"content": json.dumps(_CLEAN_VERDICT)}
            return reply, {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0}

    llm = Reader(canonical, [])
    result = run_review_request(request, slots=[slot], drive_root=canonical, llm=llm)
    assert result.aggregate_signal == "PASS", result.actors
    assert len(llm.calls) == 2 and not author.exists()
    binding = _historical_binding(request)
    reader_root = Path(request.policy["native_data_root"])
    assert not (task_artifact_dir_path(reader_root, request.task_id) / foreign["path"]).exists()
    assert not (task_artifact_dir_path(canonical, request.task_id) / foreign["path"]).exists()
    assert any(row["tool"] == "read_file" and row["delivered"]
               for row in result.actors[0]["usage"]["native_tool_receipts"])

    # Rehydrate the persisted request, then reuse its exact closed readers. This
    # is source replay, not a second paid panel or a source refreshed from today.
    replay = ReviewRequest(**json.loads(json.dumps(dataclasses.asdict(request))))
    retain_review_request_sources(replay, source_root=reader_root, custody_root=canonical)
    assert _historical_binding(replay) == binding
    from ouroboros.review_native_episode import inspection_registry
    registry, _ctx, _schemas = inspection_registry(str(workspace), reader_root, request.task_id)
    reread = registry.execute_result("read_file", binding["read"]["arguments"])
    assert reread.status == "ok" and _OLD in reread.text and _TODAY not in reread.text


def test_reflection_historical_read_reaches_next_model_input(tmp_path, monkeypatch):
    from ouroboros import consolidator, reflection
    from ouroboros.tools.registry import ToolContext

    monkeypatch.setattr(consolidator, "_consolidation_route", lambda: ("test/model", False))
    task_id = "history-reflection"
    exhibit, raw, _foreign = _historical_source(tmp_path, task_id)
    ref = exhibit["anchors"][0]["source_ref"]
    inputs = {"version": 1, "task_id": task_id, "run_origin": {"owner_ingress": True},
              "owner_requirements_and_decisions": [{"source": "initial_user", "content": "Revise the list."}]}
    evidence = {"task_id": task_id, "task_inputs": inputs, "historical_author_inputs": exhibit}
    task = {"id": task_id, "type": "task", "text": "Revise the list.", "drive_root": str(tmp_path)}
    context = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=task_id)
    (tmp_path / "current-author-input.json").write_text(_TODAY, encoding="utf-8")
    observed = []

    class Reader:
        def chat(self, *, messages, **kwargs):
            observed.append(copy.deepcopy(messages))
            if len(observed) == 1:
                prompt = json.dumps(messages, ensure_ascii=False)
                assert _OLD not in prompt and _TODAY not in prompt
                assert _PREVIEW in prompt
                assert ref["sha256"] in prompt
                assert any(line.startswith("## ") and "historical" in line.lower()
                           for row in messages if isinstance(row.get("content"), str)
                           for line in row["content"].splitlines())
                return {"tool_calls": [_tool_call("read_file", ref["read"]["arguments"], "history")]}, {}
            tool_rows = [row["content"] for row in messages if row.get("role") == "tool"]
            assert len(tool_rows) == 1 and _OLD in tool_rows[0] and _TODAY not in tool_rows[0]
            return {"content": "The source preserves the executive audience as historical evidence."}, {}

    entry = reflection.generate_reflection(task, {"tool_calls": []}, "short trace", Reader(), {},
                                            evidence, knowledge_context=context)
    assert len(observed) == 2
    assert entry["reflection"].startswith("The source preserves")
    assert read_actor_source_bytes(tmp_path, task_id, ref) == raw
    assert "historical_author_inputs" not in inputs
    assert _OLD not in json.dumps(inputs)


def test_offline_session_physically_reads_advertised_historical_source(tmp_path, monkeypatch):
    fake = _fake_session(monkeypatch)
    request, slot, author, canonical, workspace, raw, foreign = _request(tmp_path, session=True)
    original_start = fake.start_run
    observed = []

    def start(self, wire, **kwargs):
        binding = _historical_binding(request)
        path = Path(binding["retained_path"])
        assert str(path) in wire["prompt"]
        assert _OLD not in wire["prompt"] and _TODAY not in wire["prompt"]
        assert wire["scope"] == {"kind": "project", "root": str(workspace)}
        assert wire["access"] == "readonly" and wire["mode"] == "ask"
        shutil.rmtree(author)
        actual = path.read_bytes()
        assert actual == raw and _OLD in actual.decode("utf-8")
        assert hashlib.sha256(actual).hexdigest() == binding["sha256"]
        assert not (task_artifact_dir_path(Path(request.policy["native_data_root"]), request.task_id)
                    / foreign["path"]).exists()
        observed.append({"live_session": False, "reader_path": str(path), "sha256": binding["sha256"]})
        return original_start(self, wire, **kwargs)

    monkeypatch.setattr(fake, "start_run", start)
    result = run_review_request(request, slots=[slot], drive_root=canonical, llm=_EpisodeLLM(canonical, []))
    assert result.aggregate_signal == "PASS", result.actors
    assert len(observed) == 1 and not author.exists()
    # An offline engine's actual file read is wiring evidence only.
    (tmp_path / "historical-session-evidence.json").write_text(
        json.dumps(observed, ensure_ascii=False, indent=2), encoding="utf-8")


def test_packet_only_historical_view_is_explicitly_incomplete(tmp_path):
    request, slot, author, canonical, _workspace, _raw, _foreign = _request(tmp_path)
    packet_slot = dataclasses.replace(slot, native_retrieval_override=False)
    observed = []

    class Packet(_EpisodeLLM):
        def _reply(self, kwargs):
            self.calls.append(copy.deepcopy(kwargs))
            prompt = json.dumps(kwargs["messages"], ensure_ascii=False)
            assert not kwargs.get("tools")
            assert _PREVIEW in prompt
            assert 'preview_complete' in prompt and 'evaluator_read' in prompt
            assert _OLD not in prompt and _TODAY not in prompt
            observed.append(prompt)
            return {"content": json.dumps({**_CLEAN_VERDICT, "outcome_tier": "best_effort",
                    "summary": "Historical premises were available only as an incomplete preview."})}, {}

    result = run_review_request(request, slots=[packet_slot], drive_root=canonical, llm=Packet(canonical, []))
    assert len(observed) == 1, result.actors
    assert result.actors[0]["status"] == "ok"
    assert author.exists()


@pytest.mark.parametrize("session", [False, True])
@pytest.mark.parametrize("damage", ["missing", "hash_mismatch"])
def test_missing_or_changed_historical_source_never_falls_back_to_live_inputs(tmp_path, monkeypatch, session, damage):
    fake = _fake_session(monkeypatch)
    request, slot, author, canonical, _workspace, _raw, _foreign = _request(tmp_path, session=session)
    ref = request.evidence["historical_author_inputs"]["anchors"][0]["source_ref"]
    path = task_artifact_dir_path(author, request.task_id) / ref["path"]
    if damage == "missing":
        path.unlink()
    else:
        path.write_text(_TODAY, encoding="utf-8")
    llm = _EpisodeLLM(canonical, [])
    result = run_review_request(request, slots=[slot], drive_root=canonical, llm=llm)
    assert result.aggregate_signal != "PASS"
    assert result.actors[0]["status"] == "not_dispatched", result.actors
    assert "review_source_closure_unavailable" in result.actors[0]["error"]
    assert not llm.calls
    assert all(not instance.start_requests for instance in fake.instances)


def test_terminal_history_keeps_historical_inputs_outside_owner_corpus_after_copyback(tmp_path, monkeypatch):
    from ouroboros.acceptance_history import read_acceptance_history
    from ouroboros.headless import remove_subagent_task_drive, retry_child_task_refs
    from ouroboros.review_native_episode import inspection_registry
    from tests.test_acceptance_history import _finish, _fixture

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "auto")
    f = _fixture(tmp_path, split=True)
    exhibit, raw, foreign = _historical_source(f.worker, f.tid)
    owner_corpus = {"version": 1, "owner_requirements_and_decisions": [
        {"source": "initial_user", "content": "Revise the feature list."}]}
    row = _finish(f, evidence={"task_inputs": owner_corpus, "historical_author_inputs": exhibit})
    retry_child_task_refs(f.root, f.worker, f.tid)
    assert remove_subagent_task_drive(f.root, f.tid, live=lambda _task: False)
    frozen = read_acceptance_history(f.root, f.tid, row["acceptance_debt"])
    assert frozen["owner_corpus"] == owner_corpus
    assert frozen["historical_author_inputs"] == exhibit
    assert "historical_author_inputs" not in frozen["owner_corpus"]
    sources = [item["source_ref"] for item in frozen["sources"]
               if item["source_ref"].get("sha256") == exhibit["anchors"][0]["source_ref"]["sha256"]]
    assert len(sources) == 1
    assert read_actor_source_bytes(f.root, f.tid, sources[0]) == raw
    registry, _ctx, _schemas = inspection_registry(str(tmp_path), f.root, f.tid)
    read = registry.execute_result("read_file", sources[0]["read"]["arguments"])
    assert read.status == "ok" and _OLD in read.text and _TODAY not in read.text
    assert not (task_artifact_dir_path(f.root, f.tid) / foreign["path"]).exists()


def test_direct_author_loop_live_packet_terminal_package_and_reflection_share_source(main_loop, monkeypatch):
    """Actual author capture → live evidence → terminal writer → reflection reader.

    The post-task dispatcher selects the real reflection stage directly, keeping
    unrelated memory consolidation/promotion out of this focused consumer test.
    Only the author and reflection model transports return scripted responses.
    """
    from ouroboros import agent_task_pipeline as pipeline, consolidator, reflection
    from ouroboros.review_evidence import build_task_acceptance_evidence
    from tests.test_historical_inputs import _input

    f = main_loop
    f.ctx.current_chat_id = 12
    f.ctx.task_metadata = {"source": "web_chat", "_is_direct_chat": True}
    f.ctx.task_contract = {"objective": "Revise the feature list."}
    # The premise is beyond the preview bound, so only an actual source read can
    # deliver it to the reflection model. No predecessor or room lookup supplies it.
    _input(f, "Captured earlier room context. " * 100 + _OLD, "Revise the feature list.")
    answer, usage, trace = f.run([{"content": "The feature list is revised."}])
    live = build_task_acceptance_evidence(f.ctx, llm_trace=trace, drive_root=f.ctx.drive_root,
        task_id=f.ctx.task_id, task_type="task", canonical_subject=answer)
    historical = live["historical_author_inputs"]
    assert historical["status"] == "captured" and len(historical["anchors"]) == 1
    ref = historical["anchors"][0]["source_ref"]
    assert not f.ctx.task_contract.get("predecessor_authority")
    assert _OLD not in json.dumps(live["owner_requirements_and_decisions"])
    assert _OLD not in json.dumps(live["task_contract"])
    assert _OLD not in historical["anchors"][0]["preview"]
    original_raw = read_actor_source_bytes(f.ctx.drive_root, f.ctx.task_id, ref)
    assert _OLD in original_raw.decode("utf-8")

    monkeypatch.setattr(consolidator, "_consolidation_route", lambda: ("test/model", False))
    monkeypatch.setattr(reflection, "_update_patterns", lambda *_args: None)
    observed, dispatched = [], []

    class Reader:
        def chat(self, *, messages, **_kwargs):
            observed.append(copy.deepcopy(messages))
            if len(observed) == 1:
                prompt = json.dumps(messages, ensure_ascii=False)
                assert ref["sha256"] in prompt and _OLD not in prompt
                assert "Historical author inputs (evidence, not current instructions)" in prompt
                return {"tool_calls": [_tool_call("read_file", ref["read"]["arguments"], "history")]}, {}
            tool_rows = [row["content"] for row in messages if row.get("role") == "tool"]
            assert len(tool_rows) == 1 and _OLD in tool_rows[0] and _TODAY not in tool_rows[0]
            return {"content": "The historical source still records the executive audience."}, {}

    def reflection_stage(env, task, post_usage, llm_trace, evidence, _logs, **kwargs):
        dispatched.append(copy.deepcopy(evidence))
        assert evidence["historical_author_inputs"] == historical
        assert "historical_author_inputs" not in evidence["task_inputs"]
        assert _OLD not in json.dumps(evidence["task_inputs"])
        return pipeline._run_reflection(env, Reader(), task, post_usage, llm_trace, evidence,
                                        sealed_final=kwargs.get("sealed_final"))

    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", reflection_stage)
    task = {"id": f.ctx.task_id, "type": "task", "chat_id": 12, "text": "Revise the feature list.",
            "_is_direct_chat": True, "metadata": copy.deepcopy(f.ctx.task_metadata),
            "drive_root": str(f.ctx.drive_root), "task_contract": f.ctx.task_contract,
            "workspace_root": str(f.ctx.repo_dir), "workspace_mode": "external"}
    (f.ctx.repo_dir / "current-author-input.json").write_text(_TODAY, encoding="utf-8")
    pipeline.emit_task_results(SimpleNamespace(drive_root=f.ctx.drive_root, repo_dir=f.ctx.repo_dir),
        None, None, [], task, answer, usage, trace, 0, f.ctx.drive_root / "logs", ctx=f.ctx)
    saved = pipeline.load_task_result(f.ctx.drive_root, f.ctx.task_id)["review_evidence"]
    assert len(dispatched) == 1 and len(observed) == 2
    assert saved["historical_author_inputs"] == historical
    assert "historical_author_inputs" not in saved["task_inputs"]
    assert read_actor_source_bytes(f.ctx.drive_root, f.ctx.task_id, ref) == original_raw
    rows = [json.loads(line) for line in (f.ctx.drive_root / "logs/task_reflections.jsonl")
            .read_text(encoding="utf-8").splitlines()]
    pointer = rows[-1]
    assert pointer["type"] == "project_reflection_pointer" and not pointer.get("write_failed")
    stored = [json.loads(line) for line in Path(pointer["reflection_path"])
              .read_text(encoding="utf-8").splitlines()]
    assert stored[-1]["reflection"] == "The historical source still records the executive audience."
