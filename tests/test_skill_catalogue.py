"""list_skills through the registered tool, the loop's result handling and a captured SDK request; no network."""
import json

import httpx
import openai
import pytest
import yaml

from ouroboros import skill_catalogue
from ouroboros.llm import LLMClient
from ouroboros.loop_tool_execution import process_tool_results
from ouroboros.skill_loader import save_enabled, summarize_skills
from ouroboros.tool_capabilities import tool_result_limit
from ouroboros.tools.registry import ToolRegistry


@pytest.fixture
def registry(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", "")
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_kw: (True, ""))
    reg = ToolRegistry(repo_dir=repo, drive_root=data)
    reg._ctx.task_id = "catalogue-consumer"
    return reg


def _seed(registry, name, *, bucket="external", root=None,
          body="Instructions: select this skill.", **header):
    folder = (root if root is not None else registry._ctx.drive_root / "skills" / bucket) / name
    folder.mkdir(parents=True)
    manifest = {"name": name, "type": "instruction", "version": "1.0.0",
                "description": f"Purpose of {name}", "when_to_use": f"When {name} applies",
                "model_experience": f"{name} adds its playbook", **header}
    (folder / "SKILL.md").write_text("---\n" + yaml.safe_dump(manifest, allow_unicode=True)
                                     + "---\n" + body, encoding="utf-8")
    return folder


def _call(registry, args):
    """One registered call through the loop's result handling (persistence included)."""
    typed = registry.execute_result("list_skills", args)
    messages, trace = [], {"tool_calls": []}
    process_tool_results([{"fn_name": "list_skills", "tool_call_id": "catalogue-call",
        "result": typed.text, "tool_result": typed, "is_error": typed.status == "error",
        "tool_args": args, "args_for_log": args, "result_meta": {"status": typed.status}}],
        messages, trace, emit_progress=lambda _message, **_kw: None, tools=registry)
    return messages[0]["content"], trace["tool_calls"][0], typed


def _consumer(registry, args):
    content, row, typed = _call(registry, args)
    assert len(content) <= tool_result_limit("list_skills")
    assert row.get("result_partial") is not True and "FULL_RESULT_SOURCE" not in content
    # The same remote builder + SDK serializer used by Main, captured at HTTP
    # dispatch, rather than claiming a pure helper result is a physical request.
    original = [{"role": "user", "content": "Choose a skill"},
        {"role": "assistant", "tool_calls": [{"id": "catalogue-call", "type": "function",
         "function": {"name": "list_skills", "arguments": json.dumps(args)}}]},
        {"role": "tool", "tool_call_id": "catalogue-call", "content": content}]
    client = LLMClient(api_key="fixture")
    target = {"provider": "openai-compatible", "resolved_model": "fixture",
              "usage_model": "fixture", "supports_openrouter_extensions": False}
    payload = client._build_remote_kwargs(target, original, "none", 128, "auto", None, None,
                                         skip_capability_fetch=True)
    captured = []

    def respond(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "fixture", "object": "chat.completion",
            "created": 0, "model": "fixture", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": "done"}}]})

    with openai.OpenAI(api_key="fixture", base_url="https://fixture.invalid/v1", max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(respond))) as sdk:
        sdk.chat.completions.create(**payload)
    assert captured[0]["messages"][-1]["content"] == content == typed.text
    return json.loads(content), typed


def _read(registry, call):
    assert call["tool"] == "read_file"
    return registry.execute("read_file", call["arguments"])


def test_index_pages_whole_records_with_previews_to_every_name(registry):
    purpose = "Review a pull request.\n  " + "p" * 300
    sees = "Adds the review playbook. " + "s" * 300
    for i in range(40):
        _seed(registry, f"skill-{i:02}", description=purpose, when_to_use="Before merging code",
              model_experience={"what_model_sees": sees, "token_effect": "about 2k tokens when read"})
    script = registry._ctx.drive_root / "skills" / "external" / "runner"
    (script / "scripts").mkdir(parents=True)
    (script / "scripts" / "hello.py").write_text("print('hi')\n")
    (script / "SKILL.md").write_text("---\nname: runner\ndescription: Runs.\nversion: 0.1.0\n"
        "type: script\nruntime: python3\nscripts:\n  - name: hello.py\n    description: Hi.\n---\n")
    args, seen, pages, rows = {}, [], 0, {}
    while True:
        page, typed = _consumer(registry, args)
        pages += 1
        assert typed.status == "ok" and page["count"] == 41
        assert page["returned"] == len(page["skills"]) > 0 and page["offset"] == len(seen)
        seen.extend(row["name"] for row in page["skills"])
        rows.update((row["name"], row) for row in page["skills"])
        if page["next"] is None:
            break
        args = page["next"]
    assert pages > 1 and seen == sorted(seen) and len(set(seen)) == 41
    row = rows["skill-39"]
    normalized = " ".join(purpose.split())
    assert row["description"] == normalized[:200]
    assert row["what_model_sees"] == sees[:160] and row["when_to_use"] == "Before merging code"
    assert row["token_effect"] == "about 2k tokens when read"
    assert row["omitted"] == {"description": len(normalized) - 200, "what_model_sees": len(sees) - 160}
    assert row["tools"] == [] and row["load_error"] == "" and row["source"] == "external"
    assert row["enabled"] is False and row["review_status"] == "pending" and row["ready"] is False
    assert "available_for_execution" not in row and "live_loaded" not in row
    runner = rows["runner"]
    assert runner["available_for_execution"] is False and "omitted" not in runner
    # Full diagnostics, identity hashes and manifest addresses stay with the name.
    full = {"readiness", "review_gate", "tool_surfaces", "model_experience", "content_hash", "manifest"}
    assert not full & set(row)


@pytest.mark.serial
def test_tool_name_preview_counts_omissions_and_name_returns_all(registry, monkeypatch):
    from ouroboros import extension_loader
    _seed(registry, "toolkit")
    for i in range(8):
        monkeypatch.setitem(extension_loader._tools, f"ext_toolkit_t{i}", {
            "skill": "toolkit", "name": f"ext_toolkit_t{i}", "description": f"Tool {i}"})
    row = _consumer(registry, {})[0]["skills"][0]
    assert row["tools"] == [f"ext_toolkit_t{i}" for i in range(6)]
    assert row["omitted"] == {"tools": 2}
    named = _consumer(registry, {"name": "toolkit"})[0]["skills"][0]
    assert [t["name"] for t in named["tool_surfaces"]] == [f"ext_toolkit_t{i}" for i in range(8)]
    assert "schema" not in json.dumps(named["tool_surfaces"])


@pytest.mark.serial
def test_oversized_metadata_and_tool_names_fit_the_first_record(registry, monkeypatch):
    from ouroboros import extension_loader

    skill_type, version, cost = "t" * 20000, "v" * 20000, "c" * 20000
    _seed(registry, "toolkit", type=skill_type, version=version,
          model_experience={"what_model_sees": "Adds tools", "token_effect": cost})
    # Escaped characters exercise the serialized budget as well as string lengths.
    names = [f"ext_toolkit_t{i}_" + "\x00" * 20000 for i in range(8)]
    for name in names:
        monkeypatch.setitem(extension_loader._tools, name, {"skill": "toolkit", "name": name})
    page, typed = _consumer(registry, {})
    assert typed.status == "ok" and page["returned"] == 1 and page["next"] is None
    row = page["skills"][0]
    assert row["type"] == skill_type[:80] and row["version"] == version[:80]
    assert row["what_model_sees"] == "Adds tools" and row["token_effect"] == cost[:160]
    assert row["tools"] == [name[:80] for name in names[:6]]
    assert row["omitted"] == {
        "type": len(skill_type) - 80, "version": len(version) - 80,
        "token_effect": len(cost) - 160, "tools": 2,
        **{f"tools[{i}]": len(name) - 80 for i, name in enumerate(names[:6])},
    }
    content, trace, named = _call(registry, {"name": "toolkit"})
    assert named.status == "ok" and trace["result_partial"] is True
    ref = json.loads(content.split("FULL_RESULT_SOURCE_JSON=", 1)[1].splitlines()[0])
    from ouroboros.artifacts import read_actor_source_bytes
    full = json.loads(read_actor_source_bytes(registry._ctx.drive_root, "catalogue-consumer", ref))["skills"][0]
    assert full["type"] == skill_type and full["version"] == version
    assert full["model_experience"]["token_effect"] == cost
    assert [tool["name"] for tool in full["tool_surfaces"]] == names


def test_extension_rows_keep_liveness_apart_from_script_execution():
    rows = [{"name": "widget", "type": "extension", "enabled": True, "review_status": "pass",
             "readiness": {"ready": True}, "available_for_execution": False,
             "desired_live": True, "live_loaded": False, "live_reason": "load_error",
             "process": "worker", "load_error": "E" * 500}]
    record = skill_catalogue.list_skills_payload({"count": 1, "skills": rows})["skills"][0]
    assert record["desired_live"] is True and record["live_loaded"] is False
    assert record["live_reason"] == "load_error" and record["process"] == "worker"
    assert record["ready"] is True and "available_for_execution" not in record
    assert record["load_error"] == "E" * 200 and record["omitted"] == {"load_error": 300}


def test_name_returns_full_row_and_read_file_call_of_the_physical_manifest(registry):
    body = "# Code review\nStep 1: read the diff.\nINSTRUCTIONS_END\n"
    playbook = _seed(registry, "playbook", body=body,
                     model_experience={"what_model_sees": "W" * 400, "token_effect": "T" * 300})
    # Provenance tags are not buckets: a self-authored skill lives in external,
    # a markerless ClawHub folder reports source=external.
    marker = {"schema_version": 1, "origin": "self_authored", "task_id": "t1", "created_at": "c1"}
    mine = _seed(registry, "mine")
    (mine / ".self_authored.json").write_text(json.dumps(marker))
    state = registry._ctx.drive_root / "state" / "skills" / "mine"
    state.mkdir(parents=True)
    (state / "self_authored.json").write_text(json.dumps(marker))
    hub = registry._ctx.drive_root / "skills" / "clawhub" / "hubjson"
    hub.mkdir(parents=True)
    hub_text = json.dumps({"name": "hubjson", "type": "instruction", "version": "1",
                           "description": "From a JSON manifest"})
    (hub / "skill.json").write_text(hub_text)
    expected = {"playbook": ("external", "external", "SKILL.md", (playbook / "SKILL.md").read_text()),
                "mine": ("self_authored", "external", "SKILL.md", (mine / "SKILL.md").read_text()),
                "hubjson": ("external", "clawhub", "skill.json", hub_text)}
    for name, (source, bucket, file, text) in expected.items():
        page, typed = _consumer(registry, {"name": name})
        assert typed.status == "ok" and page["found"] is True and "ok" not in page
        row = page["skills"][0]
        assert (row["source"], row["location"], row["manifest_file"]) == (source, bucket, file)
        assert page["manifest"]["read"]["arguments"] == {
            "root": "skill_payload", "bucket": bucket, "skill_name": name, "path": file}
        assert set(page) == {"found", "name", "skills", "manifest"} and set(page["manifest"]) == {"read"}
        assert text in _read(registry, page["manifest"]["read"])
    # Selection and reading stay separate: the catalogue never carries the text.
    assert "INSTRUCTIONS_END" not in registry.execute("list_skills", {"name": "playbook"})
    row = _consumer(registry, {"name": "playbook"})[0]["skills"][0]
    # Full, unclipped diagnostics: the same row the shared summary carries.
    shared = next(r for r in summarize_skills(registry._ctx.drive_root)["skills"]
                  if r["name"] == "playbook")
    assert row == json.loads(json.dumps(shared))
    assert row["model_experience"] == {"what_model_sees": "W" * 400, "token_effect": "T" * 300}
    assert row["readiness"]["next_actions"] and row["review_gate"] and row["content_hash"]
    assert row["enabled"] is False  # reading instructions needs no enablement


@pytest.mark.parametrize("bucket,seeded,source", [
    ("native", True, "native"), ("native", False, "external"),
    ("user_repo", False, "user_repo"),
])
def test_native_and_user_repo_pointers_read_through_registry(
    registry, tmp_path, monkeypatch, bucket, seeded, source,
):
    from ouroboros.contracts.task_constraint import TaskConstraint

    checkout = tmp_path / "checkout"
    root = checkout / "group" if bucket == "user_repo" else None
    folder = _seed(registry, "playbook", bucket=bucket, root=root, body="EXACT_PLAYBOOK_BODY")
    if bucket == "user_repo":
        monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(checkout))
    if seeded:
        (folder / ".seed-origin").write_text("launcher-seed\n")
    manifest = (folder / "SKILL.md").read_text()

    page, typed = _consumer(registry, {"name": "playbook"})
    row = page["skills"][0]
    assert typed.status == "ok" and page["found"] is True
    assert (row["source"], row["location"], row["enabled"]) == (source, bucket, False)
    call = page["manifest"]["read"]
    assert call == {"tool": "read_file", "arguments": {
        "root": "skill_payload", "bucket": bucket, "skill_name": "playbook", "path": "SKILL.md"}}
    assert "EXACT_PLAYBOOK_BODY" not in typed.text
    read = registry.execute_result(call["tool"], call["arguments"])
    assert read.status == "ok" and manifest in read.text

    # A parent may hand this exact pointer to a read-only child. Its existing
    # file access works, while catalogue discovery and mutation remain refused.
    registry._ctx.task_constraint = TaskConstraint(mode="local_readonly_subagent")
    assert registry.get_schema_by_name("list_skills") is None
    refused = registry.execute_result("list_skills", {"name": "playbook"})
    assert refused.status != "ok" and "LOCAL_READONLY_SUBAGENT_BLOCKED" in refused.text
    child_read = registry.execute_result(call["tool"], call["arguments"])
    assert child_read.status == "ok" and manifest in child_read.text
    write = registry.execute_result("write_file", {**call["arguments"], "content": "changed"})
    assert write.status != "ok" and (folder / "SKILL.md").read_text() == manifest


def test_readable_manifest_stays_addressed_when_other_payload_is_unreadable(registry):
    folder = _seed(registry, "leaky", body="LEAKY_INSTRUCTIONS")
    (folder / ".env").write_text("TOKEN=x\n")
    page, typed = _consumer(registry, {"name": "leaky"})
    row = page["skills"][0]
    assert typed.status == "ok" and "payload unreadable" in row["load_error"]
    assert row["content_hash"] == "" and "LEAKY_INSTRUCTIONS" in _read(registry, page["manifest"]["read"])
    listed = _consumer(registry, {})[0]["skills"][0]
    assert listed["name"] == "leaky" and "payload unreadable" in listed["load_error"]


@pytest.mark.parametrize("bad_bytes", [b"\xff", b"---\nname: [unterminated\n---\n"])
def test_unreadable_manifest_is_a_failed_lookup_without_a_document(registry, bad_bytes):
    folder = _seed(registry, "broken")
    (folder / "SKILL.md").write_bytes(bad_bytes)
    page, typed = _consumer(registry, {"name": "broken"})
    assert typed.status == "error" and page["ok"] is False
    assert page["error"]["code"] == "SKILL_MANIFEST_UNREADABLE" and "manifest" not in page
    row = page["skills"][0]
    assert row["load_error"].startswith("manifest") and row["manifest_file"] == ""
    assert row["location"] == "external"
    index, listed = _consumer(registry, {})
    assert listed.status == "ok" and index["broken"] == 1
    assert index["skills"][0]["name"] == "broken" and index["skills"][0]["load_error"]


def test_collision_lists_every_candidate_and_selects_no_document(registry):
    _seed(registry, "same", body="External body")
    _seed(registry, "same", bucket="clawhub", body="Other body")
    page, typed = _consumer(registry, {"name": "same"})
    assert typed.status == "error" and page["error"]["code"] == "SKILL_IDENTITY_COLLISION"
    assert sorted(r["location"] for r in page["skills"]) == ["clawhub", "external"]
    assert "manifest" not in page and all("collision" in r["load_error"] for r in page["skills"])
    index = _consumer(registry, {})[0]
    assert [r["name"] for r in index["skills"]] == ["same", "same"]


@pytest.mark.parametrize("installed", [False, True])
def test_missing_name_completes_and_bad_arguments_are_distinct(registry, installed):
    if installed:
        _seed(registry, "present")
    content, trace_row, typed = _call(registry, {"name": "missing"})
    assert typed.status == "ok" and typed.code == "LEGACY_WARNING"
    assert trace_row["is_error"] is False and json.loads(content)["found"] is False
    for args in ({"offset": -1}, {"name": "present", "offset": 1},
                 {"name": "present", "snapshot": "x"}, {"detail": True}, {"limit": 5},
                 {"name": 1}, {"snapshot": 1}, {"offset": True}, {"offset": "1"}):
        result = registry.execute_result("list_skills", args)
        assert result.status == "error" and "TOOL_ARG_ERROR" in result.text


def test_empty_catalogue_keeps_the_install_hint(registry):
    page, typed = _consumer(registry, {})
    assert typed.status == "ok" and "ok" not in page
    assert page["skills"] == [] and page["count"] == page["returned"] == page["offset"] == 0
    assert page["next"] is None and page["snapshot"]
    assert "OUROBOROS_SKILLS_REPO_PATH" in page["hint"] and "install" in page["hint"].lower()
    stale, typed = _consumer(registry, {"snapshot": "previous-installation"})
    assert typed.status == "error" and stale["ok"] is False
    assert stale["error"]["code"] == "SKILLS_SNAPSHOT_CHANGED"


def test_snapshot_detects_membership_not_state_or_payload_edits(registry):
    _seed(registry, "a")
    b = _seed(registry, "b")
    first = _consumer(registry, {})[0]
    save_enabled(registry._ctx.drive_root, "a", True)
    with (b / "SKILL.md").open("a") as stream:
        stream.write("\nNew instructions")
    same, typed = _consumer(registry, {"snapshot": first["snapshot"]})
    assert typed.status == "ok" and same["snapshot"] == first["snapshot"]
    assert same["skills"][0]["enabled"] is True
    _seed(registry, "c")
    added, typed = _consumer(registry, {"offset": 1, "snapshot": first["snapshot"]})
    assert typed.status == "error" and added["ok"] is False and added["skills"] == []
    assert added["error"]["code"] == "SKILLS_SNAPSHOT_CHANGED"
    with_c = added["snapshot"]
    (b / "SKILL.md").unlink()
    removed = _consumer(registry, {"snapshot": with_c})[0]
    assert removed["error"]["code"] == "SKILLS_SNAPSHOT_CHANGED"


def test_oversized_named_detail_rides_the_generic_result_source(registry):
    trigger = "w" * 20000 + " WHEN_END"
    _seed(registry, "long", when_to_use=trigger)
    compact = _consumer(registry, {})[0]["skills"][0]
    assert compact["when_to_use"] == trigger[:160]
    assert compact["omitted"]["when_to_use"] == len(trigger) - 160
    content, trace_row, typed = _call(registry, {"name": "long"})
    assert typed.status == "ok" and len(typed.text) > tool_result_limit("list_skills")
    assert trace_row["result_partial"] is True and trace_row["result_source_status"] == "ready"
    ref = json.loads(content.split("FULL_RESULT_SOURCE_JSON=", 1)[1].splitlines()[0])
    tail = registry.execute(ref["read"]["tool"], {**ref["read"]["arguments"], "start_char": 19000})
    assert "WHEN_END" in tail
    from ouroboros.artifacts import read_actor_source_bytes
    exact = json.loads(read_actor_source_bytes(registry._ctx.drive_root, "catalogue-consumer", ref))
    assert exact["skills"][0]["when_to_use"] == trigger
    assert exact["manifest"]["read"]["arguments"]["path"] == "SKILL.md"
