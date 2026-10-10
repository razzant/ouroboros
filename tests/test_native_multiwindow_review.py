"""One native review keeps exact coverage across actor-owned working views."""

import ast
import copy
import dataclasses
import hashlib
import json

import pytest

from ouroboros import review_native_episode as native
from ouroboros.artifacts import read_actor_source_bytes
from tests.test_native_tool_round_executor import _assignment, _tool_call, _ScriptedLLM


def _source(path, root="system_repo"):
    raw = path.read_bytes()
    text = raw.decode().replace("\r\n", "\n").replace("\r", "\n")
    return {"root": root, "path": path.name, "source_revision": hashlib.sha256(raw).hexdigest(),
            "complete_sha256": hashlib.sha256(text.encode()).hexdigest(), "complete_chars": len(text),
            "range_basis": "unicode_text_universal_newlines"}


def _native(repo, data, llm, sources):
    assignment = dataclasses.replace(_assignment(repo, llm), custody_root=data)
    assignment.request.policy["native_required_sources"] = sources
    return native.NativeToolRoundReviewExecutor(assignment, llm=llm)


class ContractReviewer:
    """Compute the finding from received source; forget producer state after focus."""

    def __init__(self):
        self.stage, self.buffer, self.cursor = "writer", "", 0
        self.calls = []
        self.findings = None

    def chat(self, **kwargs):
        messages = kwargs["messages"]
        self.calls.append(copy.deepcopy(kwargs))
        tools = [m for m in messages if m.get("role") == "tool"]
        response = tools[-1]["content"] if tools else None
        if self.stage in {"writer", "reader"} and response is not None and self.cursor:
            body = response.partition("\n")[2].partition("\n⚠️ RESULT TRUNCATED:")[0]
            assert body
            self.buffer += body
            if "RESULT TRUNCATED:" not in response:
                tree = ast.parse(self.buffer)
                namespace = {}
                exec(compile(tree, "fixture", "exec"), namespace, namespace)
                if self.stage == "writer":
                    table = ast.literal_eval(tree.body[0].value)
                    # The uniform contract is computed over every actual route.
                    encoded = {namespace["encode"](route, 90)["ttl"] for route in table}
                    assert len(encoded) == 1
                    self.note = "I checked the producer for every supported route; next verify the consumer. Producer evidence: " + json.dumps(
                        {"seconds": 90, "ttl": encoded.pop(), "routes": len(table)})
                    self.buffer, self.cursor, self.stage = "", 0, "inspect"
                    return self._call("compact_context", {"inspect": True})
                note_text = next(b["text"] for m in messages for b in m.get("content", [])
                                 if isinstance(b, dict) and (b.get("_context_capsule") or {}).get("authorship") == "actor")
                witness = json.loads(note_text.split("Producer evidence: ", 1)[1])
                table = ast.literal_eval(tree.body[0].value)
                assert len(table) == witness["routes"]
                self.findings = []
                for route in table:
                    decoded = namespace["decode"]({"route": route, "ttl": witness["ttl"]})
                    if decoded != witness["seconds"]:
                        self.findings.append({"severity": "critical", "item": "ttl_roundtrip", "evidence":
                            f"writer.py encodes {witness['seconds']} seconds as {witness['ttl']}; reader.py decodes {route} as {decoded} seconds",
                            "recommendation": "Use matching producer and consumer units."})
                return {"content": json.dumps(self.findings)}, {"cost": 0.0}
        if self.stage == "inspect":
            # This actor only needs the revision to replace all completed units.
            # The richer unit/source inventory may be a disclosed partial JSON
            # view under this deliberately tiny 8K result cap. Decode the exact
            # visible revision field, not the incomplete object or unseen units.
            revision, _ = json.JSONDecoder().raw_decode(
                response.split('"view_revision":', 1)[1].lstrip())
            self.stage = "apply"
            note, self.note = self.note, None
            return self._call("compact_context", {"expected_view_revision": revision,
                                                    "working_note": note, "keep_unit_ids": []})
        if self.stage == "apply":
            # No producer source, table or witness remains in the fake model.
            assert self.buffer == "" and self.note is None
            assert any("Producer evidence:" in str(m.get("content")) for m in messages)
            self.stage, self.cursor = "reader", 0
            response = None
        offset = len(self.buffer)
        self.cursor += 1
        return self._call("read_file", {"path": f"{self.stage}.py", "start_line": 1,
                                        "max_lines": 3, "start_char": offset})

    def _call(self, name, args):
        return {"tool_calls": [_tool_call(name, args, f"c{len(self.calls)}")]}, {"cost": 0.0}


@pytest.mark.parametrize("broken", [True, False], ids=["cross-file-unit-defect", "clean-control"])
def test_native_review_reads_beyond_one_window_and_computes_cross_file_finding(tmp_path, monkeypatch, broken):
    repo, data = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    table = {f"route_{i:04d}": 1000 for i in range(3800)}
    writer = "TO_WIRE = " + repr(table) + "\ndef encode(route, seconds): return {'route': route, 'ttl': round(seconds * TO_WIRE[route])}\n"
    if broken:
        table["route_2137"] = 1
    reader = "FROM_WIRE = " + repr(table) + "\ndef decode(message): return message['ttl'] / FROM_WIRE[message['route']]\n"
    (repo / "writer.py").write_text(writer)
    (repo / "reader.py").write_text(reader)
    bound = 120_000
    assert len(writer) + len(reader) > bound
    monkeypatch.setattr(native, "review_native_transcript_bound", lambda *a, **k: bound)
    monkeypatch.setattr(native, "_EPISODE_TOOL_RESULT_CHAR_CAP", 8000)
    llm = ContractReviewer()
    executor = _native(repo, data, llm, [_source(repo / "writer.py"), _source(repo / "reader.py")])
    result = executor.execute()
    findings = json.loads(result.raw_text)
    assert bool(findings) is broken
    if broken:
        assert len(findings) == 1 and "route_2137" in findings[0]["evidence"] and "90000" in findings[0]["evidence"]
    assert result.usage["native_view_changes"] == 1
    assert result.usage["native_read_coverage"]["status"] == "complete"
    assert "native_incomplete" not in result.message
    sizes = [native._wire_size(call["messages"], call["tools"]) for call in llm.calls]
    assert max(sizes) <= bound and any(b < a / 2 for a, b in zip(sizes, sizes[1:]))
    assert any(r.get("text_chars", 0) > 0 and r["end_line"] < r["start_line"] for r in executor._tool_receipts)
    history = json.loads(read_actor_source_bytes(data, "t-native", result.usage["native_history_source"]))
    assert len(history["round_sources"]) == len(llm.calls)
    assert history["coverage"]["status"] == "complete"
    assert all(row["status"] == "complete" for row in history["coverage"]["sources"])
    for ref in history["round_sources"]:
        assert json.loads(read_actor_source_bytes(data, "t-native", ref))["messages"]


def test_more_than_200_observed_reads_close_the_required_source(tmp_path):
    repo, data = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    (repo / "part.txt").write_text("first\nsecond\n")
    calls = [{"tool_calls": [_tool_call("read_file", {"path": "part.txt", "max_lines": 1}, f"c{i}")]}
             for i in range(200)]
    calls += [{"tool_calls": [_tool_call("read_file", {"path": "part.txt", "start_line": 2}, "last")]},
              {"content": "[]"}]
    executor = _native(repo, data, _ScriptedLLM(calls), [_source(repo / "part.txt")])
    result = executor.execute()
    assert len(result.usage["native_tool_receipts"]) == 201
    assert result.usage["native_read_coverage"]["status"] == "complete"
    history = json.loads(read_actor_source_bytes(data, "t-native", result.usage["native_history_source"]))
    assert len(history["read_receipts"]) == 201 and history["read_receipts"][-1]["delivered"]


def test_same_normalized_text_on_a_new_raw_revision_cannot_complete_old_coverage(tmp_path):
    repo, data = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    path = repo / "source.txt"
    path.write_bytes(b"first\r\nsecond\r\n")
    required = [_source(path)]

    class RewritingReviewer(_ScriptedLLM):
        def chat(self, **kwargs):
            if len(self.calls) == 1:
                path.write_bytes(b"first\nsecond\n")
            return super().chat(**kwargs)

    llm = RewritingReviewer([
        {"tool_calls": [_tool_call("read_file", {"path": path.name, "max_lines": 1}, "prefix")]},
        {"tool_calls": [_tool_call("read_file", {"path": path.name}, "new-whole")]}, {"content": "[]"}])
    result = _native(repo, data, llm, required).execute()
    row = result.usage["native_read_coverage"]["sources"][0]
    assert row["status"] == "incomplete" and row["missing_ranges"] == [[6, 13]]
    assert result.raw_text == "[]" and result.message["native_incomplete"] == "required_source_coverage_incomplete"


def test_read_not_carried_by_another_physical_send_does_not_count_as_delivered(tmp_path):
    repo, data = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    (repo / "file.txt").write_text("source\n")

    class FailsBeforeSend(_ScriptedLLM):
        def chat(self, **kwargs):
            if self.calls:
                raise RuntimeError("pre-dispatch failure")
            return super().chat(**kwargs)

    llm = FailsBeforeSend([{"tool_calls": [_tool_call("read_file", {"path": "file.txt"})]}])
    executor = _native(repo, data, llm, [_source(repo / "file.txt")])
    with pytest.raises(RuntimeError):
        executor.execute()
    usage = executor.failure_custody()
    assert usage["native_tool_receipts"][0]["delivered"] is False
    assert usage["native_read_coverage"]["status"] == "incomplete"


def test_two_part_brief_states_the_read_provenance_each_delivery_produces(tmp_path):
    """The brief's pre-run manifest states WHICH provenance this seat's receipts
    will carry — host-executed for a native episode, parsed from the harness
    journal for a delegated session — instead of the blanket `unobserved` that
    described neither. It is never a claim that a source WAS read: that is the
    post-run coverage fact. (Window size still never rewrites a finding; that
    invariant is pinned in tests/test_review_brief_coupling.py.)"""
    from ouroboros.tools import review_brief_coupling as brief_mod
    from ouroboros.tools.scope_required_sources import required_sources_ref
    path = tmp_path / "source.py"
    path.write_text("def actual_behavior(): return 1\n", encoding="utf-8")
    required = [_source(path)]
    ref = required_sources_ref(required)
    task, manifest = brief_mod.build_retrieving_brief(tmp_path, brief_mod.BriefInputs(
        commit_message="Review the change",
        intent=brief_mod.BriefIntent(goal="Check the actual behavior"),
        required_sources=required, required_sources_ref=ref))
    assert manifest["native_required_sources"] == required
    assert manifest["native_required_sources_ref"] == ref
    assert manifest["read_provenance_expected"] == "host_observed"
    assert "host_file_read_attestation" not in manifest
    assert ref["sha256"] in task and "independent from your working window" in task

    _task, delegated = brief_mod.build_retrieving_brief(tmp_path, brief_mod.BriefInputs(
        commit_message="Review the change", delegated=True,
        required_sources=required, required_sources_ref=ref))
    assert delegated["read_provenance_expected"] == "harness_observed"


def test_native_legacy_compaction_argument_requests_an_authored_note_explicitly(tmp_path):
    repo, data = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    llm = _ScriptedLLM([{"tool_calls": [_tool_call("compact_context", {"keep_last_n": 2})]}, {"content": "[]"}])
    executor = _native(repo, data, llm, [])
    executor.execute()
    text = next(m["content"] for m in llm.calls[1]["messages"] if m.get("role") == "tool")
    assert "your own working_note" in text
    assert getattr(executor._inspection_ctx, "_pending_compaction", None) is None


def test_a_declared_empty_manifest_is_complete_coverage_of_nothing(tmp_path):
    """The four coverage states must be distinguishable. A surface that declared
    an EMPTY manifest (this change owes the reviewer no source in full) is
    covered, not unobserved: nothing was required and nothing is missing."""
    repo, data = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    executor = _native(repo, data, _ScriptedLLM([{"content": "[]"}]), [])
    result = executor.execute()
    coverage = result.usage["native_read_coverage"]
    assert coverage["status"] == "complete" and coverage["reason"] == "declared_empty"
    assert coverage["required_source_count"] == 0
    assert "native_incomplete" not in result.usage


def test_no_declared_manifest_at_all_stays_unobserved(tmp_path):
    """An absent manifest key is a provenance limit on what may be CLAIMED about
    coverage, not a finding that the review was incomplete (BIBLE P3)."""
    import dataclasses

    from tests.test_native_tool_round_executor import _assignment

    repo, data = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    llm = _ScriptedLLM([{"content": "[]"}])
    assignment = dataclasses.replace(_assignment(repo, llm), custody_root=data)
    assignment.request.policy.pop("native_required_sources", None)
    result = native.NativeToolRoundReviewExecutor(assignment, llm=llm).execute()
    coverage = result.usage["native_read_coverage"]
    assert coverage["status"] == "unobserved"
    assert coverage["reason"] == "required_source_manifest_missing"
    assert "native_incomplete" not in result.usage


@pytest.mark.parametrize("inline", [False, True])
def test_source_delivered_inline_needs_no_second_read(tmp_path, inline):
    repo, data = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    path = repo / "owed.txt"
    path.write_text("required body\n", encoding="utf-8")
    row = {**_source(path), "coverage_basis": "delivered_inline" if inline else "candidate_blob"}
    llm = _ScriptedLLM([{"content": "[]"}])
    executor = _native(repo, data, llm, [row])
    if inline:
        executor.assignment.request.session_task += "\n" + path.read_text(encoding="utf-8")
    result = executor.execute()
    coverage = result.usage["native_read_coverage"]
    assert coverage["status"] == ("complete" if inline else "incomplete")
    assert coverage["sources"][0]["missing_ranges"] == ([] if inline else [[0, 14]])
    assert coverage["sources"][0]["covered_chars"] == (14 if inline else 0)
    assert ("native_incomplete" in result.usage) is not inline
    assert result.raw_text == "[]" and len(llm.calls) == 1
    assert result.usage["host_file_read_attestation"] == "host_observed"
    assert result.usage["native_tool_receipts"] == []
