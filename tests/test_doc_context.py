"""Tests for the low/max reference-doc layout in the always-on agent context.

Doc matrix (D-ARCH unification, owner 2026-08-08):
  ARCHITECTURE follows the OWNER MODE alone: full in max for EVERY task class
       (self-body, project tasks with or without a folder, evolution, external
       surfaces); navigation map in low. It is Ouroboros's capability/tools/
       access map — never dropped per-task in max.
  DEVELOPMENT is MODE-INDEPENDENT per task class, keyed by D-DEV (owner
       2026-08-08) on the ACTIVE REPO BINDING — the handbook loads exactly when
       the work targets Ouroboros's own body, a path fact and never a guess from
       message text (P5). On-demand pointer for the EXTERNAL-SURFACE class (a
       bound workspace incl. an auto-provisioned genesis tree, a subagent, an
       api/cli/scheduled surface); full for everything still bound to the system
       repo — including a direct-chat turn in a PROJECT ROOM, which binds no
       workspace. Project MEMBERSHIP is deliberately NOT the signal. Explicit
       context_requires_development / context_requires_self_body_docs win.
  README/CHECKLISTS: on-demand pointer in all modes.
SYSTEM + BIBLE are tier-0 and always full.
"""

import os
import pathlib
import tempfile
import pytest
from tests._governance_docs_shared import governance_doc_text
from tests.test_swarm_host_admission import host as _promotion_host

promotion_host = _promotion_host

# Unique sentinel placed inside the ARCHITECTURE body so we can prove the full
# body is inlined (max) vs replaced by a structure-only nav map (low).
_ARCH_BODY_SENTINEL = "ARCH_BODY_SENTINEL_XYZ"


def _make_env_and_memory(tmpdir: pathlib.Path):
    from ouroboros.agent import Env
    from ouroboros.memory import Memory

    repo_dir = tmpdir / "repo"
    drive_root = tmpdir / "drive"
    repo_dir.mkdir(parents=True, exist_ok=True)
    drive_root.mkdir(parents=True, exist_ok=True)
    for subdir in ["state", "memory", "memory/knowledge", "logs"]:
        (drive_root / subdir).mkdir(parents=True, exist_ok=True)
    (repo_dir / "prompts").mkdir(parents=True, exist_ok=True)
    (repo_dir / "docs").mkdir(parents=True, exist_ok=True)
    (repo_dir / "prompts" / "SYSTEM.md").write_text("You are Ouroboros.", encoding="utf-8")
    (repo_dir / "BIBLE.md").write_text("# Principle 0: Agency", encoding="utf-8")
    (repo_dir / "docs" / "ARCHITECTURE.md").write_text(
        "# Ouroboros v5.5.0 — Architecture\n\n## Section A\n\n"
        + _ARCH_BODY_SENTINEL
        + " alpha\n\n### Section A child\n\n#### Section A detail\n\ndetail\n\n"
        + "## Section B\n\nbeta\n",
        encoding="utf-8",
    )
    (repo_dir / "docs" / "DEVELOPMENT.md").write_text("# DEVELOPMENT.md — Dev Guide", encoding="utf-8")
    (repo_dir / "README.md").write_text('[![Version 5.5.0](https://img.shields.io/badge/version-5.5.0-green.svg)](VERSION)', encoding="utf-8")
    (repo_dir / "docs" / "CHECKLISTS.md").write_text("## Change Review Checklist\n| # | item |", encoding="utf-8")
    (drive_root / "state" / "state.json").write_text('{"spent_usd": 0}', encoding="utf-8")
    (drive_root / "memory" / "scratchpad.md").write_text("test scratchpad", encoding="utf-8")
    (drive_root / "memory" / "identity.md").write_text("I am Ouroboros.", encoding="utf-8")
    env = Env(repo_dir=repo_dir, drive_root=drive_root)
    memory = Memory(drive_root=drive_root, repo_dir=repo_dir)
    return env, memory


def _build_system_text(task_overrides=None, *, context_mode="max"):
    from ouroboros.context import build_llm_messages
    tmpdir = pathlib.Path(tempfile.mkdtemp())
    env, memory = _make_env_and_memory(tmpdir)
    task = {"id": "test-1", "type": "task", "text": "hello"}
    if task_overrides:
        task.update(task_overrides)
    prev = os.environ.get("OUROBOROS_CONTEXT_MODE")
    os.environ["OUROBOROS_CONTEXT_MODE"] = context_mode
    try:
        messages, _ = build_llm_messages(env=env, memory=memory, task=task)
    finally:
        if prev is None:
            os.environ.pop("OUROBOROS_CONTEXT_MODE", None)
        else:
            os.environ["OUROBOROS_CONTEXT_MODE"] = prev
    content = messages[0]["content"]
    return " ".join(block.get("text", "") for block in content if isinstance(block, dict))


def test_max_mode_inlines_architecture_and_development_in_full():
    text = _build_system_text(context_mode="max")
    assert "## ARCHITECTURE.md" in text
    assert _ARCH_BODY_SENTINEL in text  # full body inlined
    assert "navigation map" not in text
    assert "## DEVELOPMENT.md" in text


def test_plan_review_docs_pin_fail_closed_exact_artifact_custody():
    repo = pathlib.Path(__file__).resolve().parents[1]

    for relative in ("docs/ARCHITECTURE.md", "docs/DEVELOPMENT.md"):
        text = governance_doc_text(relative, repo)
        assert "plan_review_exact_artifact_unavailable" in text, relative


def test_forked_task_captures_canonical_memory_and_exact_api_context():
    import json

    from ouroboros.context import build_llm_messages
    from ouroboros.contracts.task_contract import attach_task_contract
    from ouroboros.memory import Memory

    tmpdir = pathlib.Path(tempfile.mkdtemp())
    env, canonical_memory = _make_env_and_memory(tmpdir)
    child = tmpdir / "child"
    (child / "logs").mkdir(parents=True)
    (child / "memory").mkdir(parents=True)
    canonical_memory.logs_path("chat.jsonl").write_text(
        '{"chat_id": 1, "direction": "in", "text": "CANONICAL_OWNER_DIRECTIVE"}\n',
        encoding="utf-8",
    )
    (child / "logs" / "chat.jsonl").write_text(
        '{"chat_id": 1, "direction": "in", "text": "CHILD_ONLY_NOISE"}\n',
        encoding="utf-8",
    )
    task = attach_task_contract({
        "id": "forked-root",
        "type": "task",
        "text": "continue",
        "context": "never deploy; use profile X",
        "drive_root": str(child),
        "budget_drive_root": str(env.drive_root),
    })
    messages, _ = build_llm_messages(
        env=env,
        memory=Memory(child, repo_dir=env.repo_dir),
        task=task,
    )
    rendered = json.dumps(messages, ensure_ascii=False)

    assert "CANONICAL_OWNER_DIRECTIVE" in rendered
    assert "CHILD_ONLY_NOISE" not in rendered
    assert "never deploy; use profile X" in rendered
    assert task["task_contract"]["context"] == "never deploy; use profile X"


def test_forked_context_uses_canonical_global_cognition_not_child_noise():
    import json

    from ouroboros.agent import Env
    from ouroboros.context import build_llm_messages
    from ouroboros.memory import Memory

    tmpdir = pathlib.Path(tempfile.mkdtemp())
    canonical_env, _memory = _make_env_and_memory(tmpdir)
    canonical = canonical_env.drive_root
    child = tmpdir / "child-cognition"
    for root in (canonical, child):
        (root / "memory" / "knowledge").mkdir(parents=True, exist_ok=True)
        (root / "logs").mkdir(parents=True, exist_ok=True)
        (root / "state").mkdir(parents=True, exist_ok=True)
    (canonical / "memory" / "knowledge" / "patterns.md").write_text(
        "CANONICAL_PATTERN_REGISTER", encoding="utf-8",
    )
    (child / "memory" / "knowledge" / "patterns.md").write_text(
        "CHILD_PATTERN_NOISE", encoding="utf-8",
    )
    (canonical / "memory" / "deep_review.md").write_text(
        "CANONICAL_DEEP_REVIEW", encoding="utf-8",
    )
    (child / "memory" / "deep_review.md").write_text("CHILD_DEEP_NOISE", encoding="utf-8")
    env = Env(repo_dir=canonical_env.repo_dir, drive_root=child, budget_drive_root=canonical)

    messages, _ = build_llm_messages(
        env=env,
        memory=Memory(child, repo_dir=env.repo_dir),
        task={"id": "fork-cognition", "text": "continue", "budget_drive_root": str(canonical)},
    )
    rendered = json.dumps(messages, ensure_ascii=False)

    assert "CANONICAL_PATTERN_REGISTER" in rendered
    assert "CANONICAL_DEEP_REVIEW" in rendered
    assert "CHILD_PATTERN_NOISE" not in rendered
    assert "CHILD_DEEP_NOISE" not in rendered


def test_deep_review_is_historical_without_shrinking_its_existing_excerpt(tmp_path):
    import json

    from ouroboros.context import build_llm_messages
    from ouroboros.utils import truncate_review_artifact

    env, memory = _make_env_and_memory(tmp_path)
    path = env.drive_root / "memory" / "deep_review.md"
    for prefix in ("LEGACY REPORT\n", "<!-- deep-review provenance: generated_at=2026-01-01T00:00:00Z, source_revision=unknown -->\n"):
        original = prefix + "retained historical reasoning " * 400
        path.write_text(original, encoding="utf-8")
        # A recent file mtime is not a review date, even on a copied old report.
        os.utime(path, (1900000000, 1900000000))
        messages, _ = build_llm_messages(
            env=env, memory=memory, task={"id": "history", "type": "task", "text": "continue"},
        )
        rendered = json.dumps(messages, ensure_ascii=False)
        assert json.dumps(truncate_review_artifact(original, limit=8000), ensure_ascii=False)[1:-1] in rendered
        assert "not a verdict on the current tree" in rendered
        assert "unrecorded date or source revision is unknown" in rendered
        assert path.read_text(encoding="utf-8") == original


def test_current_plan_and_open_dispositions_enter_actual_model_request():
    import json

    from ouroboros.context import build_llm_messages

    tmpdir = pathlib.Path(tempfile.mkdtemp())
    env, memory = _make_env_and_memory(tmpdir)
    task_id = "reviewed-task"
    fingerprint = "a" * 64
    result_dir = env.drive_root / "task_results"
    result_dir.mkdir(parents=True)
    (result_dir / f"{task_id}.json").write_text(json.dumps({
        "_schema_version": 1,
        "task_id": task_id,
        "status": "running",
        "plan_review_state": {
            "schema_version": 2,
            "series_id": "series-1",
            "cycles_paid": 1,
            "need_evidence_seen": [],
            "current_attempt": {
                "fingerprint": fingerprint,
                "status": "open",
                "reason": "",
            },
            "waves": [{
                "cycle_index": 1,
                "request_fingerprint": fingerprint,
                "aggregate": "REVISE_PLAN",
                "closed": False,
                "paid": True,
                "spec": {"prose": "PLAN_DECISIVE_TAIL"},
                "findings": [{"id": "F1", "summary": "OPEN_FINDING_TAIL"}],
                "dispositions": [{"finding_id": "F1", "decision": "open"}],
            }],
            "waves_omitted": 0,
        },
    }), encoding="utf-8")

    messages, _ = build_llm_messages(
        env=env,
        memory=memory,
        task={"id": task_id, "type": "task", "text": "continue"},
    )
    rendered = json.dumps(messages, ensure_ascii=False)

    assert "PLAN_DECISIVE_TAIL" in rendered
    assert "OPEN_FINDING_TAIL" in rendered
    assert "get_task_result" in rendered


def test_valid_named_predecessor_materializes_into_first_model_request_and_actor_read():
    import json

    from ouroboros.agent_startup_checks import validate_task_authority_sources
    from ouroboros.context import build_llm_messages
    from ouroboros.tools.control import _get_task_result
    from ouroboros.tools.registry import ToolContext

    tmpdir = pathlib.Path(tempfile.mkdtemp())
    env, memory = _make_env_and_memory(tmpdir)
    predecessor_id = "cat-predecessor"
    tail = "CLAUDEXOR_ONLY_AND_L1_L2_L3"
    result_dir = env.drive_root / "task_results"
    result_dir.mkdir(parents=True)
    (result_dir / f"{predecessor_id}.json").write_text(json.dumps({
        "_schema_version": 1,
        "task_id": predecessor_id,
        "status": "completed",
        "objective": "o" * 700 + tail,
        "task_contract": {
            "objective": "o" * 700 + tail,
            "context": "never use native API",
            "delegation_budget": {"intent_note": "L1 asks L2 to spawn L3"},
        },
    }), encoding="utf-8")
    source = {
        "kind": "task_result", "task_id": predecessor_id,
        "human_label": "Cat Tower Builder",
        "tool": "get_task_result",
        "arguments": {"task_id": predecessor_id, "include_authority": True},
    }
    task = {
        "id": "cat-next", "type": "task", "text": "continue",
        "predecessor_authority_source": source,
        "budget_drive_root": str(env.drive_root),
    }

    assert validate_task_authority_sources(env, task) == {}
    messages, _ = build_llm_messages(env=env, memory=memory, task=task)
    rendered = json.dumps(messages, ensure_ascii=False)
    actor_read = json.loads(_get_task_result(
        ToolContext(repo_dir=env.repo_dir, drive_root=env.drive_root),
        predecessor_id, include_authority=True,
    ))

    assert tail in rendered
    assert "L1 asks L2 to spawn L3" in rendered
    assert actor_read["status"] == "available"
    assert actor_read["authority"]["task_contract"]["objective"].endswith(tail)


def test_unreadable_current_plan_review_source_returns_typed_startup_refusal():
    from ouroboros.agent_startup_checks import validate_task_authority_sources

    tmpdir = pathlib.Path(tempfile.mkdtemp())
    env, _memory = _make_env_and_memory(tmpdir)
    result_dir = env.drive_root / "task_results"
    result_dir.mkdir(parents=True)
    (result_dir / "plan-broken.json").write_text("{broken", encoding="utf-8")

    refusal = validate_task_authority_sources(
        env, {"id": "plan-broken", "title": "Cat plan", "text": "continue"},
    )

    assert refusal["reason_code"] == "authority_source_unavailable"
    assert refusal["source"]["kind"] == "plan_review_state"
    assert refusal["human_label"] == "Cat plan"


def test_named_owner_source_resolves_beyond_automatic_recent_generations():
    import json

    from ouroboros.agent_startup_checks import validate_task_authority_sources
    from ouroboros.project_dialogue import build_owner_message_ref

    tmpdir = pathlib.Path(tempfile.mkdtemp())
    env, _memory = _make_env_and_memory(tmpdir)
    archive = env.drive_root / "archive"
    logs = env.drive_root / "logs"
    archive.mkdir(parents=True)
    logs.mkdir(parents=True, exist_ok=True)
    text = "old but exact owner authority"
    ref = build_owner_message_ref(
        chat_id=1, client_message_id="old-owner", ts="2026-08-01T00:00:00Z", text=text,
    )
    (archive / "chat_0001.jsonl").write_text(
        json.dumps({**ref, "direction": "in", "text": text}) + "\n", encoding="utf-8",
    )
    for index in range(2, 6):
        (archive / f"chat_000{index}.jsonl").write_text(
            json.dumps({"direction": "in", "text": f"newer-{index}"}) + "\n",
            encoding="utf-8",
        )
    (logs / "chat.jsonl").write_text(
        json.dumps({"direction": "in", "text": "live"}) + "\n", encoding="utf-8",
    )
    task = {"id": "legacy-source", "origin_message_ref": ref}

    assert validate_task_authority_sources(env, task) == {}
    assert task["origin_message_text"] == text


def test_max_mode_external_workspace_keeps_arch_full_but_drops_development():
    """D-ARCH: ARCHITECTURE is full-resident in max for EVERY class, including
    the external-surface class; DEVELOPMENT (the self-engineering handbook)
    is the on-demand pointer there — external work targets OTHER codebases."""
    from ouroboros.contracts.task_contract import build_task_contract

    external = _build_system_text(
        {
            "workspace_root": "/tmp/example-workspace",
            "workspace_mode": "external",
            "actor_id": "cli",
            "metadata": {"source": "cli"},
        },
        context_mode="max",
    )
    assert "## ARCHITECTURE.md" in external
    assert _ARCH_BODY_SENTINEL in external  # capability map stays resident
    assert "navigation map" not in external
    assert "## DEVELOPMENT.md" not in external  # handbook is the pointer
    assert "DEVELOPMENT.md" in external  # ...but visibly named (P1)

    self_body = _build_system_text(
        {
            "workspace_root": "/tmp/example-workspace",
            "workspace_mode": "external",
            "actor_id": "cli",
            "metadata": {"source": "cli"},
            "context_requires_self_body_docs": True,
        },
        context_mode="max",
    )
    assert _ARCH_BODY_SENTINEL in self_body
    assert "## DEVELOPMENT.md" in self_body  # explicit self-body keeps DEV full

    contract = build_task_contract({
        "id": "task-docs",
        "context_requires_self_body_docs": "true",
        "metadata": {"source": "api_task"},
    })

    assert contract["context_requires_self_body_docs"] is True

    contract_false = build_task_contract({
        "id": "task-docs-false",
        "context_requires_self_body_docs": "false",
        "metadata": {"source": "api_task"},
    })
    assert contract_false["context_requires_self_body_docs"] is False


def test_low_mode_external_workspace_gets_both_book_navigation_views():
    external = _build_system_text(
        {
            "workspace_root": "/tmp/example-workspace",
            "workspace_mode": "external",
            "actor_id": "cli",
            "metadata": {"source": "cli"},
        },
        context_mode="low",
    )
    assert "navigation map" in external
    assert _ARCH_BODY_SENTINEL not in external
    assert "## DEVELOPMENT.md (navigation map)" in external


@pytest.mark.serial
@pytest.mark.parametrize("mode", ["max", "low", "nano"])
def test_promoted_self_body_requirement_reaches_the_contract_and_context_without_becoming_project_state(
    promotion_host, tmp_path, monkeypatch, mode,
):
    """Schema → tool → event handler → real queue/contract → root and child context."""
    import copy
    from types import SimpleNamespace
    from ouroboros.agent import Env
    from ouroboros.context import build_llm_messages
    from ouroboros.memory import Memory
    from ouroboros.projects_registry import create_project, get_project
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.control import _build_child_subagent_contract, get_tools
    from supervisor.events import _handle_promote_chat_to_task

    host = promotion_host
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", mode)
    monkeypatch.setattr("ouroboros.context.resolve_context_fit_route", lambda *_a, **_kw: (
        {"model": "fixture", "provider": "fixture"},
        SimpleNamespace(route_fp="fixture", status="asserted", stale=False, window_tokens=1_000_000),
    ))
    books_env, _memory = _make_env_and_memory(tmp_path / "books")
    env = Env(repo_dir=books_env.repo_dir, drive_root=host.root, budget_drive_root=host.root)
    development_body = "# DEVELOPMENT.md — Dev Guide\n\n## Review\n\nRead every source before committing.\n"
    (env.repo_dir / "docs" / "DEVELOPMENT.md").write_text(development_body, encoding="utf-8", newline="\n")
    memory = Memory(host.root, repo_dir=env.repo_dir)
    workspace = tmp_path / "working-copy"
    workspace.mkdir()
    project = create_project(host.root, "code-copy", name="Code Copy")
    entry = next(tool for tool in get_tools() if tool.name == "promote_chat_to_task")
    field = "context_requires_self_body_docs"
    parameters = entry.schema["parameters"]
    assert parameters["properties"][field]["type"] == "boolean"
    assert field not in parameters["required"]
    description = parameters["properties"][field]["description"]
    assert "Ouroboros code" in description and "copies elsewhere" in description
    assert "context_requires_development" not in parameters["properties"]
    events = []

    def emit(event):
        events.append(copy.deepcopy(event))
        _handle_promote_chat_to_task(event, host.ctx)

    tool_ctx = SimpleNamespace(
        repo_dir=env.repo_dir, system_repo_dir=env.repo_dir, drive_root=host.root, task_id="promoter",
        current_chat_id=1, task_metadata={}, pending_events=[], event_queue=SimpleNamespace(put_nowait=emit),
    )
    first = None
    # A later task in the SAME project must not inherit the first task's choice.
    for flag in (True, None, False):
        kwargs = {} if flag is None else {field: flag}
        result = entry.handler(tool_ctx, "Inspect this working copy", predecessor_task_id="",
                               project_id=project["id"], workspace_root=str(workspace), **kwargs)
        assert result.startswith("OK: task"), result
        task = copy.deepcopy(host.pending[-1])
        assert events[-1][field] is bool(flag)
        assert task["task_contract"][field] is bool(flag)
        assert task[field] is bool(flag)
        assert pathlib.Path(task["workspace_root"]).resolve() == workspace.resolve()
        assert load_task_result(host.root, task["id"])["task_contract"][field] is bool(flag)
        assert field not in get_project(host.root, project["id"])
        # Exercise the existing contract-only consumer, as on restored tasks.
        task.pop(field)
        messages, _cap = build_llm_messages(env=env, memory=memory, task=task)
        content = messages[0]["content"]
        first = first or content[0]
        assert content[0] == first and "docs/DEVELOPMENT.md" in content[0]["text"]
        handbook = [block for block in content if block["text"].startswith("## DEVELOPMENT.md\n")]
        assert len(handbook) == int(bool(flag) and mode == "max")
        if handbook:
            assert content[1] == handbook[0] and "cache_control" in handbook[0]
            assert development_body in handbook[0]["text"]
        if mode != "max":
            assert "## DEVELOPMENT.md (navigation map)" in content[0]["text"]
        child = {"id": "child-" + task["id"], "type": "task", "text": "Read the assignment", "chat_id": task["chat_id"],
                 "delegation_role": "subagent", "parent_task_id": task["id"], "root_task_id": task["id"],
                 "configured_subagent": {"id": "helper", "route": {"kind": "api_model"}}}
        child["task_contract"] = _build_child_subagent_contract({
            "tid": child["id"], "objective": child["text"], "parent_contract": task["task_contract"],
            "parent_task_id": task["id"], "root_task_id": task["id"], "workspace_root": str(workspace),
            "workspace_mode": "external", "parent_project_id": project["id"],
        })
        assert child["task_contract"][field] is bool(flag)
        child_messages, _cap = build_llm_messages(env=env, memory=memory, task=child)
        child_content = child_messages[0]["content"]
        assert "## DEVELOPMENT.md (navigation map)" in child_content[0]["text"]
        assert not any(block["text"].startswith("## DEVELOPMENT.md\n") for block in child_content)
    assert len(events) == len(host.pending) == 3 and not host.attempts


def test_max_mode_evolution_task_keeps_arch_and_development_full():
    """D-ARCH (owner, 2026-08-08): evolution gets the FULL capability map in
    max ('и эволюции тоже её класть нужно') — the v6.30.0 nav-map downgrade is
    removed — and keeps the engineering handbook inline."""
    text = _build_system_text({"type": "evolution"}, context_mode="max")
    assert "## ARCHITECTURE.md" in text
    assert _ARCH_BODY_SENTINEL in text
    assert "navigation map" not in text
    assert "## DEVELOPMENT.md" in text  # handbook still full

    # Deep self-review keeps the full self-body docs (unchanged).
    review_text = _build_system_text({"type": "deep_self_review"}, context_mode="max")
    assert _ARCH_BODY_SENTINEL in review_text

    # Low retains navigation to both books; Max's full bodies stay unchanged.
    low_text = _build_system_text({"type": "evolution"}, context_mode="low")
    assert "navigation map" in low_text
    assert _ARCH_BODY_SENTINEL not in low_text
    assert "## DEVELOPMENT.md" in low_text


def test_development_keys_on_the_repo_binding_not_project_membership():
    """D-DEV (owner 2026-08-08): the handbook loads iff the work targets
    Ouroboros's own body, and the structural signal is the ACTIVE REPO BINDING —
    NOT `project_id`. ARCHITECTURE is unaffected and follows the owner mode alone
    (D-ARCH): full in max for every class, navigation map in low."""
    # A project task with a BOUND workspace (this is what a promoted project task
    # is since Q10=A auto-provisions a genesis tree) works on ANOTHER codebase.
    folder_max = _build_system_text(
        {"project_id": "proj_sub", "workspace_root": "/tmp/proj-tree", "workspace_mode": "external"},
        context_mode="max",
    )
    assert _ARCH_BODY_SENTINEL in folder_max  # ARCH full in max, always
    assert "## DEVELOPMENT.md" not in folder_max
    assert "DEVELOPMENT.md" in folder_max  # named in the on-demand pointer

    # A direct-chat turn in a PROJECT ROOM binds no workspace: still Ouroboros's
    # own body, so it KEEPS the handbook. This is the case the project_id-keyed
    # draft got wrong.
    room_chat_max = _build_system_text(
        {"project_id": "proj_sub", "_is_direct_chat": True}, context_mode="max"
    )
    assert _ARCH_BODY_SENTINEL in room_chat_max
    assert "## DEVELOPMENT.md" in room_chat_max
    room_chat_low = _build_system_text(
        {"project_id": "proj_sub", "_is_direct_chat": True}, context_mode="low"
    )
    assert "navigation map" in room_chat_low
    assert _ARCH_BODY_SENTINEL not in room_chat_low
    assert "## DEVELOPMENT.md" in room_chat_low

    # workspace="none" binds nothing -> the task is not external -> keeps it.
    opt_out = _build_system_text(
        {"project_id": "proj_sub", "workspace": "none"}, context_mode="max"
    )
    assert "## DEVELOPMENT.md" in opt_out

    # Evolution / self-body keep it through the self-body branch even in a room.
    evolution = _build_system_text(
        {"project_id": "proj_sub", "type": "evolution"}, context_mode="max"
    )
    assert "## DEVELOPMENT.md" in evolution

    # Explicit per-task overrides still win in BOTH directions.
    explicit_dev = _build_system_text(
        {
            "project_id": "proj_sub", "workspace_root": "/tmp/proj-tree",
            "workspace_mode": "external", "context_requires_development": True,
        },
        context_mode="max",
    )
    assert "## DEVELOPMENT.md" in explicit_dev
    explicit_self_body = _build_system_text(
        {
            "project_id": "proj_sub", "workspace_root": "/tmp/proj-tree",
            "workspace_mode": "external", "context_requires_self_body_docs": True,
        },
        context_mode="max",
    )
    assert "## DEVELOPMENT.md" in explicit_self_body
    explicit_off = _build_system_text(
        {"project_id": "proj_sub", "context_requires_development": False}, context_mode="max"
    )
    assert "## DEVELOPMENT.md" not in explicit_off


def test_readme_and_checklists_are_on_demand_pointer_in_both_modes():
    for mode in ("max", "low"):
        text = _build_system_text(context_mode=mode)
        assert "Reference docs available on demand" in text
        # Named in the pointer (visible, never silently dropped) but not inlined.
        assert "README.md" in text
        assert "CHECKLISTS.md" in text


def test_low_mode_architecture_is_navigation_map_not_full_body():
    text = _build_system_text(context_mode="low")
    assert "navigation map" in text
    assert "Section A" in text and "Section B" in text  # headings present
    assert "- Section A — lines 3-12" in text  # parent keeps its complete subtree
    assert "  - Section A child — lines 7-12" in text
    assert "    - Section A detail — lines 9-12" in text
    assert _ARCH_BODY_SENTINEL not in text  # full body NOT inlined in low


def test_low_mode_development_navigation_for_task_and_direct_chat():
    code_text = _build_system_text({"type": "task"}, context_mode="low")
    assert "## DEVELOPMENT.md (navigation map)" in code_text

    chat_text = _build_system_text({"_is_direct_chat": True}, context_mode="low")
    assert "## DEVELOPMENT.md (navigation map)" in chat_text

    pure_chat_text = _build_system_text(
        {"_is_direct_chat": True, "context_requires_development": False},
        context_mode="low",
    )
    assert "## DEVELOPMENT.md (navigation map)" in pure_chat_text


# Predicted route pressure no longer changes the document projection. The
# complete behavioral matrix lives in test_context_fit_v664; this guards the
# deletion seam from acquiring a compatibility shim.
def test_predicted_route_downgrade_authority_is_absent():
    from ouroboros import loop

    assert not hasattr(loop, "_maybe_downgrade_max_unconfirmed")


def test_an_empty_seed_is_a_blank_drive_not_a_blank_context():
    """#1321: memory_mode=empty seeds nothing onto the child's own drive, yet the
    child's context is still the canonical governance and shared memory -- the
    schema says exactly that instead of "starts blank"."""
    import json

    from ouroboros.context import build_llm_messages
    from ouroboros.headless import prepare_task_drive
    from ouroboros.memory import Memory
    from ouroboros.tools.control_subagent_spec import schedule_subagent_properties

    tmpdir = pathlib.Path(tempfile.mkdtemp())
    env, canonical_memory = _make_env_and_memory(tmpdir)
    canonical_memory.logs_path("chat.jsonl").write_text(
        '{"chat_id": 1, "direction": "in", "text": "CANONICAL_SHARED_DIALOGUE"}\n', encoding="utf-8")
    child = prepare_task_drive(env.drive_root, "emptychild1", "empty")
    assert child is not None and not (child / "memory" / "identity.md").exists()
    forked = prepare_task_drive(env.drive_root, "forkedchild1", "forked")
    assert (forked / "memory" / "identity.md").read_text(encoding="utf-8") == "I am Ouroboros."
    task = {"id": "emptychild1", "type": "task", "text": "verify", "parent_task_id": "root1",
            "root_task_id": "root1", "delegation_role": "subagent", "memory_mode": "empty",
            "drive_root": str(child), "budget_drive_root": str(env.drive_root)}
    messages, _ = build_llm_messages(env=env, memory=Memory(child, repo_dir=env.repo_dir), task=task)
    rendered = json.dumps(messages, ensure_ascii=False)
    assert "I am Ouroboros." in rendered and "Principle 0: Agency" in rendered
    description = schedule_subagent_properties()["memory_mode"]["description"]
    assert "starts blank" not in description and "not a blank context" in description
    assert "canonical governance" in description and "empty seeds that drive with nothing" in description
