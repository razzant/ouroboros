"""The rolling working checkpoint stores each repeated element once: parts, manifest, exact restore."""
from __future__ import annotations

import copy
import hashlib
import json
import pathlib
from types import SimpleNamespace

import pytest

from ouroboros import evidence_packs as packs
from ouroboros import observability as obs
from ouroboros import working_checkpoint as wc
from ouroboros.observability import _json_bytes
from ouroboros.request_wire_contract import physical_candidate_bytes
from ouroboros.shared_values import SharedValueMissing, token_digests
from ouroboros.task_results import write_task_result
from tests._budget_pause_exact_helpers import _loop_ctx
from tests._cases import chosen_rows

SECRET = "ghp_" + "abcdefghijklmnopqrstuvwxyz123456"


def _transcript(rounds: int, start: int = 0) -> list:
    messages = [{"role": "system", "content": "SYSTEM Ω " * 500}, {"role": "user", "content": "Inspect the project."}]
    for index in range(start, start + rounds):
        call = f"call_{index}"
        messages += [
            {"role": "assistant", "content": f"Round {index}", "tool_calls": [
                {"id": call, "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": f"m{index}.py"})}}]},
            {"role": "tool", "tool_call_id": call, "content": f"File {index}: café / состояние / 😀\n" + "x" * 8000},
        ]
    return messages


def _state_fixture(root, task_id="pause-task", rounds=30):
    ctx, limit_ctx = _loop_ctx(root, task_id)
    limit_ctx.messages = _transcript(rounds)
    limit_ctx.llm_trace = {"tool_calls": [{"id": f"call_{i}", "name": "read_file", "status": "success", "result": "ok " * 50}
                                          for i in range(rounds)]}
    limit_ctx.tool_schemas = [{"type": "function", "function": {"name": f"t{i}", "parameters": {"type": "object"}}} for i in range(12)]
    ctx._owner_directives = [{"text": "keep the audit order", "msg_id": "m1"}, {"text": "cite the receipt", "msg_id": "m2"}]
    ctx._last_context_observation = {"messages": copy.deepcopy(limit_ctx.messages), "tool_schemas": copy.deepcopy(limit_ctx.tool_schemas)}
    limit_ctx.accumulated_usage["llm_call_refs"] = [{"call_id": f"llm_{i}", "manifest_ref": {"path": f"/x/{i}.json"}} for i in range(rounds)]
    return ctx, limit_ctx


def _files(root: pathlib.Path) -> dict:
    return {p: p.stat().st_size for p in root.rglob("*") if p.is_file() and not p.name.startswith(".")}


def _manifest(root, task_id="pause-task", attempt=1) -> dict:
    path = obs.call_manifest_path(root, task_id, f"working-a{attempt}")
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _file(root, task_id="pause-task", attempt=1):
    return wc.checkpoint_path(root, task_id, attempt)


@pytest.fixture(autouse=True)
def _clean():
    packs._READERS.clear()
    yield
    packs._READERS.clear()


def test_a_checkpoint_is_a_skeleton_over_cas_parts_listed_by_one_manifest_per_attempt(tmp_path):
    ctx, limit_ctx = _state_fixture(tmp_path)
    before = _files(tmp_path)
    assert wc.save_round(limit_ctx, "post_batch") is True
    written = {p: size for p, size in _files(tmp_path).items() if p not in before}
    data = _file(tmp_path).read_bytes()
    wrapper = json.loads(data)
    assert wrapper["format"] == wc.WORKING_SHARED_FORMAT and set(wrapper) == {"format", "state"}
    skeleton = wrapper["state"]
    assert skeleton["task_id"] == "pause-task", "the skeleton keeps every scalar fact inline"
    digests = set(token_digests(data))
    elements = (limit_ctx.messages + limit_ctx.tool_schemas + limit_ctx.llm_trace["tool_calls"] + ctx._owner_directives
                + limit_ctx.accumulated_usage["llm_call_refs"])
    expected = {hashlib.sha256(_json_bytes(e)).hexdigest() for e in elements}
    assert digests == expected, "messages, tool schemas, trace calls, directives, call refs and the observation copies"
    assert all(set(token_digests(json.dumps(skeleton["context_observations"]["_last_context_observation"]).encode()))
               <= expected for _ in [0]), "the observation's copies name the same parts as the state's own messages"
    manifest = _manifest(tmp_path)
    assert manifest["call_type"] == wc.WORKING_PARTS_CALL_TYPE and manifest["task_attempt"] == 1
    assert {p["sha256"] for p in manifest["shared_parts"]} == expected
    for part in manifest["shared_parts"]:
        raw = packs.blob_bytes(tmp_path, part)
        assert hashlib.sha256(raw).hexdigest() == part["sha256"] and raw == _json_bytes(json.loads(raw))
    stats = limit_ctx.accumulated_usage["working_checkpoint"]
    assert stats["parts_written"] == len(expected) and stats["last_bytes"] == len(data)
    assert stats["last_whole_bytes"] == len(wc.read_working(tmp_path, "pause-task", 1)[0])
    whole = json.dumps(wc.read_working(tmp_path, "pause-task", 1)[1], ensure_ascii=False).encode()
    assert len(data) * 10 < len(whole), f"the file is {len(data)} B for a {len(whole)} B state"
    assert sum(written.values()) <= len(whole) * 1.2, "the first save writes each element once plus the skeleton and the manifest"

    # The next boundary: one new tool result and one new trace call -> two parts, a tiny file, one manifest rewrite.
    limit_ctx.messages.append({"role": "tool", "tool_call_id": "call_29", "content": "done again"})
    limit_ctx.llm_trace["tool_calls"].append({"id": "call_30", "name": "read_file", "status": "success", "result": "late"})
    before = _files(tmp_path)
    assert wc.save_round(limit_ctx, "ready") is True
    written = {p: size for p, size in _files(tmp_path).items() if p not in before or before[p] != size}
    assert stats["parts_written"] == len(expected) + 2
    assert sum(written.values()) * 10 < stats["last_whole_bytes"], f"the second save wrote {sum(written.values())} B"
    assert len(_manifest(tmp_path)["shared_parts"]) == len(expected) + 2, "the manifest is the union of the attempt's parts"
    assert sorted(p.name for p in _file(tmp_path).parent.glob("working_checkpoint-a*")) == ["working_checkpoint-a1.json"]


@pytest.mark.parametrize("boundary", chosen_rows([*wc.BOUNDARIES, wc.RESUMED], 0, 2, 4))
def test_restore_at_every_boundary_yields_the_same_continuation_state(tmp_path, boundary):
    from ouroboros.owner_wait import continuation_state

    ctx, limit_ctx = _state_fixture(tmp_path, rounds=8)
    write_task_result(tmp_path, "pause-task", "running")
    expected = json.loads(json.dumps(continuation_state(
        ctx, limit_ctx.messages, limit_ctx.llm_trace,
        {k: v for k, v in limit_ctx.accumulated_usage.items() if k not in wc._TRANSIENT_USAGE},
        limit_ctx.round_idx, limit_ctx.tool_schemas, set(limit_ctx.owner_msg_seen)), ensure_ascii=False))
    if boundary == wc.RESUMED:
        expected["context_start"] = ["unknown", ""]  # this legacy continuation carried no original-start fact
        wc.save_resumed(limit_ctx.tools, {"round_idx": 4, "tool_schemas": limit_ctx.tool_schemas,
                                          "resume_point": {"budget_tail": "tool"}, "context_model_role": ""},
                        limit_ctx.messages, limit_ctx.llm_trace, limit_ctx.accumulated_usage, limit_ctx.owner_msg_seen)
    else:
        assert wc.save_round(limit_ctx, boundary) is True
    data, state = wc.read_working(tmp_path, "pause-task", 1)
    assert state["working"]["boundary"] == boundary
    restored = {k: v for k, v in state.items() if k != "working"}
    expected.pop("usage"); restored.pop("usage")  # the writer adds its own counters there
    assert restored == expected, "value equality of the whole continuation state"
    assert physical_candidate_bytes({"messages": state["messages"], "tools": state["tool_schemas"]}) == \
        physical_candidate_bytes({"messages": limit_ctx.messages, "tools": limit_ctx.tool_schemas})
    assert json.loads(data) == state and b'"$shared"' not in data, "the frozen bytes are whole: no token survives expansion"
    # The recovery freezes the expanded state; the consumer loads it without the CAS.
    task: dict = {"id": "pause-task"}
    assert wc.attach_recovery(tmp_path, task, source_task_id="pause-task", from_attempt=1, cause="worker_crash")
    from ouroboros.artifacts import read_actor_source_bytes

    frozen = read_actor_source_bytes(tmp_path, "pause-task", task["_working_recovery"]["source_ref"])
    assert frozen == data and json.loads(frozen)["messages"] == limit_ctx.messages
    for part in _manifest(tmp_path)["shared_parts"]:
        (tmp_path / "observability" / "blobs" / f"{part['sha256']}.json.gz").unlink()
    consumer = SimpleNamespace(task_id="pause-task", task_attempt=2, drive_root=tmp_path, budget_drive_root=tmp_path)
    loaded = wc.load_recovery(consumer, task["_working_recovery"])
    assert loaded["messages"] == limit_ctx.messages and loaded["trace"] == limit_ctx.llm_trace


def test_private_opaque_and_foreign_elements_stay_inline(tmp_path):
    ctx, limit_ctx = _state_fixture(tmp_path, rounds=3)
    from ouroboros.anthropic_native_custody import ANTHROPIC_NATIVE_RECEIPT_KEY

    secret_row = {"role": "tool", "tool_call_id": "call_s", "content": f"api_key={SECRET}"}
    receipt_row = {"role": "assistant", "content": "native", ANTHROPIC_NATIVE_RECEIPT_KEY: {"content_json": "[]"}}
    opaque_row = {"role": "tool", "tool_call_id": "call_o", "content": "x", "handle": object()}
    limit_ctx.messages += [secret_row, receipt_row, opaque_row]
    assert wc.save_round(limit_ctx, "post_batch") is True
    data = _file(tmp_path).read_bytes()
    skeleton = json.loads(data)["state"]
    inline = [row for row in skeleton["messages"] if "$shared" not in row]
    assert [row.get("tool_call_id") or row.get("content") for row in inline] == ["call_s", "native", "call_o"]
    assert SECRET.encode() in data and SECRET not in "".join(
        packs.blob_bytes(tmp_path, p).decode() for p in _manifest(tmp_path)["shared_parts"]), "no secret enters the CAS"
    assert inline[2]["handle"] == {"__unserializable__": "object"}
    _data, state = wc.read_working(tmp_path, "pause-task", 1)
    assert state["messages"][-3] == secret_row and state["messages"][-2] == receipt_row
    assert state["working"]["opaque_values"] == ["object"]
    # A token-shaped dict inside an element is data inside its part; one at a dict position outside every
    # element would make expansion ambiguous, so that state is written whole, and both read back exactly.
    limit_ctx.messages.append({"role": "user", "content": {"$shared": "a" * 64}})
    assert wc.save_round(limit_ctx, "ready") is True
    assert json.loads(_file(tmp_path).read_bytes())["format"] == wc.WORKING_SHARED_FORMAT
    assert wc.read_working(tmp_path, "pause-task", 1)[1]["messages"][-1]["content"] == {"$shared": "a" * 64}
    limit_ctx.accumulated_usage["odd"] = {"$shared": "b" * 64}
    limit_ctx.messages.append({"role": "user", "content": "again"})
    assert wc.save_round(limit_ctx, "ready") is True
    plain = json.loads(_file(tmp_path).read_bytes())
    assert plain.get("format") is None and plain["task_id"] == "pause-task" and plain["usage"]["odd"] == {"$shared": "b" * 64}
    assert wc.read_working(tmp_path, "pause-task", 1)[1]["messages"][-1] == {"role": "user", "content": "again"}


def test_the_checkpoint_and_the_request_records_share_each_parts_redaction_fact(tmp_path):
    """A part the round's request records already worked out is not scanned again by the checkpoint (and the other
    way round): one fact per bytes per task (``observability.task_parts``). What the checkpoint skips is exactly
    what the request scanned; a secret-bearing message stays inline whichever met it first."""
    from ouroboros.context_timing import measured_sections, section_spans

    def scanned(work) -> int:
        with measured_sections():
            work()
            return section_spans()["work"].get("secret_scan", 0)

    ctx, limit_ctx = _state_fixture(tmp_path, rounds=4)
    limit_ctx.messages.append({"role": "tool", "tool_call_id": "call_s", "content": f"api_key={SECRET}"})
    request = {"messages": limit_ctx.messages, "send_messages": limit_ctx.messages, "tools": limit_ctx.tool_schemas}
    obs._TASK_PARTS.clear()
    by_request = scanned(lambda: obs.persist_call(tmp_path, task_id="pause-task", call_id="llm_1_request",
                                                  call_type="llm_request", payload=request))
    after_request = scanned(lambda: wc.save_round(limit_ctx, "post_batch"))
    obs._TASK_PARTS.clear()
    alone = scanned(lambda: wc.save_round(limit_ctx, "post_batch"))
    assert by_request > 0 and alone - after_request == by_request
    inline = [row for row in json.loads(_file(tmp_path).read_bytes())["state"]["messages"] if "$shared" not in row]
    assert [row.get("tool_call_id") for row in inline] == ["call_s"]


def test_the_manifest_union_survives_a_resume_in_a_new_process_and_a_dropped_element_stays_listed(tmp_path):
    ctx, limit_ctx = _state_fixture(tmp_path, rounds=4)
    assert wc.save_round(limit_ctx, "post_batch") is True
    first = {p["sha256"] for p in _manifest(tmp_path)["shared_parts"]}
    created = _manifest(tmp_path)["created_at"]
    # Compaction replaces the transcript: the new process (fresh ctx memo) lists only what is new, keeps the old.
    ctx2, limit2 = _state_fixture(tmp_path, rounds=4)
    limit2.messages = [limit2.messages[0], {"role": "assistant", "content": "[compacted] rounds 0-3"}] + _transcript(1, start=50)[2:]
    ctx2._last_context_observation = {"messages": copy.deepcopy(limit2.messages), "tool_schemas": []}
    assert wc.save_round(limit2, "ready") is True
    manifest = _manifest(tmp_path)
    listed = {p["sha256"] for p in manifest["shared_parts"]}
    assert first < listed and manifest["created_at"] == created, "a part an earlier state named stays listed; the manifest keeps its birth"
    dropped = first - set(token_digests(_file(tmp_path).read_bytes()))
    assert dropped, "the compacted transcript no longer names the old rounds"
    for digest in dropped:  # still reachable: listed by the attempt's manifest, retired with the task by the packer
        assert (tmp_path / "observability" / "blobs" / f"{digest}.json.gz").is_file()
    assert set(token_digests(_file(tmp_path).read_bytes())) <= listed
    assert {sha for _kind, sha, _size in packs.manifest_blob_refs(tmp_path, "pause-task")} == listed, "the packer's walk sees every part"


@pytest.mark.parametrize("failure", ["part", "manifest"])
def test_a_manifest_write_that_failed_is_written_by_the_same_contexts_next_save(tmp_path, monkeypatch, failure):
    """A part counts as listed only once a manifest naming it landed: the save that retries after a failed
    manifest write (``save_or_log``'s next boundary, ``_hold_until``'s next try) lists it before its file names it."""
    _ctx, limit_ctx = _state_fixture(tmp_path, rounds=4)
    real, calls = obs.write_call_manifest, []
    publish = obs.replace_atomic
    part_attempts = []

    def part_fails_once(source, target):
        part_attempts.append(target)
        if failure == "part" and len(part_attempts) == 2:
            raise OSError("the disk refused a part")
        return publish(source, target)

    def fails_once(*args, **kwargs):
        calls.append(kwargs.get("call_id"))
        if failure == "manifest" and len(calls) == 1:
            raise OSError("the disk refused the manifest")
        return real(*args, **kwargs)

    monkeypatch.setattr(obs, "write_call_manifest", fails_once)
    monkeypatch.setattr(obs, "replace_atomic", part_fails_once)
    with pytest.raises(OSError):
        wc.save_round(limit_ctx, "post_batch")
    assert not _file(tmp_path).exists() and _manifest(tmp_path) == {}, "no file names a part no manifest lists"
    assert wc.save_round(limit_ctx, "post_batch") is True
    named = set(token_digests(_file(tmp_path).read_bytes()))
    listed = {p["sha256"] for p in _manifest(tmp_path)["shared_parts"]}
    assert calls == (["working-a1"] * (2 if failure == "manifest" else 1)) and named and named <= listed
    assert {sha for _kind, sha, _size in packs.manifest_blob_refs(tmp_path, "pause-task")} == listed, "the packer's walk sees every part"


def test_an_older_body_refuses_the_shared_form_typed_and_a_missing_part_is_typed(tmp_path):
    ctx, limit_ctx = _state_fixture(tmp_path, rounds=2)
    write_task_result(tmp_path, "pause-task", "running")
    assert wc.save_round(limit_ctx, "candidate") is True
    data = _file(tmp_path).read_bytes()

    def older_read_working(root, task_id, attempt):  # the reader every body before this change ran
        state = json.loads(wc.checkpoint_path(root, task_id, attempt).read_bytes().decode("utf-8"))
        if (not isinstance(state, dict) or state.get("task_id") != str(task_id)
                or int(state.get("task_attempt") or 0) != int(attempt) or not isinstance(state.get("working"), dict)):
            raise ValueError("working checkpoint identity mismatch")
        return data, state

    with pytest.raises(ValueError, match="identity mismatch"):
        older_read_working(tmp_path, "pause-task", 1)
    # This body: a part nothing holds is a typed refusal; recovery keeps the original-prompt path, the file stays.
    part = _manifest(tmp_path)["shared_parts"][0]
    (tmp_path / "observability" / "blobs" / f"{part['sha256']}.json.gz").unlink()
    with pytest.raises(SharedValueMissing, match=part["sha256"]):
        wc.read_working(tmp_path, "pause-task", 1)
    task: dict = {"id": "pause-task"}
    assert not wc.attach_recovery(tmp_path, task, source_task_id="pause-task", from_attempt=1, cause="worker_crash")
    assert "_working_recovery" not in task and _file(tmp_path).read_bytes() == data
    # A plain state written by an older body still reads.
    _file(tmp_path).write_bytes(json.dumps(
        {"task_id": "pause-task", "task_attempt": 1, "working": {"boundary": "ready", "seq": 1}, "messages": []}).encode())
    assert wc.read_working(tmp_path, "pause-task", 1)[1]["messages"] == []


def test_a_pausing_seed_and_a_park_source_are_completed_from_the_expanded_state(tmp_path):
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.budget_pause import STATE_PAUSING

    ctx, limit_ctx = _state_fixture(tmp_path, rounds=3)
    write_task_result(tmp_path, "pause-task", "running")
    assert wc.save_round(limit_ctx, "post_batch") is True
    from ouroboros.budget_pause import set_budget_pause

    row = {"pause_id": "p1", "state": STATE_PAUSING, "reason": "budget", "rail": "global_exhausted",
           "pause_generation": 1, "scope": "global", "task_attempt": 1, "source_ref": None}
    set_budget_pause(tmp_path, "pause-task", row)
    completed = wc.complete_pause_from_working(tmp_path, "pause-task", row, 1)
    source = json.loads(read_actor_source_bytes(tmp_path, "pause-task", completed["source_ref"]))
    assert source["messages"] == limit_ctx.messages and "$shared" not in json.dumps(source)


def test_the_packer_retires_the_parts_of_a_settled_task_and_keeps_a_live_tasks(tmp_path, monkeypatch):
    from ouroboros import evidence_packer as packer
    from ouroboros import store

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    ctx, limit_ctx = _state_fixture(tmp_path, task_id="settled", rounds=3)
    assert wc.save_round(limit_ctx, "post_batch") is True
    ctx2, limit2 = _state_fixture(tmp_path, task_id="live", rounds=3)
    assert wc.save_round(limit2, "post_batch") is True
    from ouroboros.utils import utc_now_iso

    with store.open_store(tmp_path, write=True) as txn:
        txn.store_task({"task_id": "settled", "status": "completed", "_schema_version": 1, "updated_at": utc_now_iso()})
        txn.store_task({"task_id": "live", "status": "running", "_schema_version": 1, "updated_at": utc_now_iso()})
    settled_parts = {p["sha256"] for p in _manifest(tmp_path, "settled")["shared_parts"]}
    live_parts = {p["sha256"] for p in _manifest(tmp_path, "live")["shared_parts"]}
    report = packer.run_pass(tmp_path)
    assert report["packed"] == 1 and report["kept"] == len(settled_parts & live_parts)
    members = {m["sha256"] for m in packs.published_generations(tmp_path, "settled")[0]["members"]}
    assert settled_parts <= members
    for digest in settled_parts - live_parts:
        assert not (tmp_path / "observability" / "blobs" / f"{digest}.json.gz").exists()
    assert wc.read_working(tmp_path, "live", 1)[1]["messages"] == limit2.messages, "the live checkpoint reads: loose or packed parts"
    assert wc.read_working(tmp_path, "settled", 1)[1]["messages"] == limit_ctx.messages
