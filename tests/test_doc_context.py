"""Tests for the reference-doc layout in the always-on agent context.

Doc matrix (owner 25.11.2026, BIBLE P1; supersedes D-ARCH and D-DEV of 2026-08-08):
  Every mode (Max, Low, Nano) and every focus carries ONE resident Body Map,
       rendered from the books' sources (`reference_books.body_map`): the
       Architecture entry's introduction, each chapter's title, physical path
       and authored opening, the Reference Atlas and the Development chapter
       list. No chapter body is preloaded in any mode.
  DEVELOPMENT is whole only as the task's own requirement
       (`task_contract.context_requires_development is True`), in any mode, as
       its own block. No binding, surface, task type or
       `context_requires_self_body_docs` preloads it.
  A single-file (legacy) Architecture revision has no Map form: its source
       navigation rides instead. Where no book renders, the prompt names the
       books, README and CHECKLISTS as on-demand reads.
SYSTEM + BIBLE are tier-0 and always full.
"""

import os
import pathlib
import tempfile
import pytest
from tests.test_swarm_host_admission import host as _promotion_host
from tests._store_shared import seed_result, unreadable_result

promotion_host = _promotion_host

# Unique sentinel placed inside the ARCHITECTURE body so we can prove the full
# body is inlined (max) vs replaced by a structure-only nav map (low).
_ARCH_BODY_SENTINEL = "ARCH_BODY_SENTINEL_XYZ"
_DEV_BODY_SENTINEL = "DEV_BODY_SENTINEL_XYZ"
_BOOKS = {
    "docs/ARCHITECTURE.md": (
        "# Ouroboros v5.5.0 — Architecture\n\nThe map of my body.\n\n## Chapters\n\n"
        "- [Section A](architecture/01-section-a.md)\n- [Section B](architecture/02-section-b.md)\n\n"
        "## Reference Atlas\n\n- [DEVELOPMENT.md](DEVELOPMENT.md) — how a change is made.\n"
        "- [CHECKLISTS.md](CHECKLISTS.md) — the questions my reviewers answer.\n"),
    "docs/architecture/01-section-a.md": (
        "# Section A\n\nWhat section A gives me.\n\n## Mechanism\n\n" + _ARCH_BODY_SENTINEL + " alpha\n"),
    "docs/architecture/02-section-b.md": "# Section B\n\nWhat section B gives me.\n\n## Detail\n\nbeta\n",
    "docs/DEVELOPMENT.md": "# DEVELOPMENT.md — Dev Guide\n\nHow I change.\n\n## Chapters\n\n- [Review](development/01-review.md)\n",
    "docs/development/01-review.md": "# Review\n\nHow a change is reviewed.\n\n## Rules\n\n" + _DEV_BODY_SENTINEL + "\n",
}
_MAP_ROWS = (
    "## Body map", "The map of my body.",
    "### Section A\nSource: `docs/architecture/01-section-a.md`\nWhat section A gives me.",
    "### Section B\nSource: `docs/architecture/02-section-b.md`\nWhat section B gives me.",
    "### Reference Atlas\n- `docs/DEVELOPMENT.md` — how a change is made.",
    "### Development handbook chapters\n- Review: `docs/development/01-review.md`",
)
HANDBOOK = "## DEVELOPMENT.md\n\n"
REQUIRED = {"task_contract": {"context_requires_development": True}}


def _map_only(text):
    """The resident Map, whole, with no chapter body, no handbook block, no line range and no version."""
    for row in _MAP_ROWS:
        assert row in text, row
    assert _ARCH_BODY_SENTINEL not in text and _DEV_BODY_SENTINEL not in text and HANDBOOK not in text
    assert "## ARCHITECTURE.md" not in text and "v5.5.0" not in text and " — lines " not in text


def _required(text):
    """The Map stays, and the task's own requirement adds the whole handbook as its block."""
    for row in _MAP_ROWS:
        assert row in text, row
    assert _ARCH_BODY_SENTINEL not in text and text.count(HANDBOOK) == 1 and _DEV_BODY_SENTINEL in text


def _make_env_and_memory(tmpdir: pathlib.Path, *, legacy: bool = False):
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
    if legacy:  # a single-file revision: no Map form, its source navigation stands in
        (repo_dir / "docs" / "ARCHITECTURE.md").write_text(
            "# Ouroboros v5.5.0 — Architecture\n\n## Section A\n\n"
            + _ARCH_BODY_SENTINEL
            + " alpha\n\n### Section A child\n\n#### Section A detail\n\ndetail\n\n"
            + "## Section B\n\nbeta\n",
            encoding="utf-8",
        )
        (repo_dir / "docs" / "DEVELOPMENT.md").write_text("# DEVELOPMENT.md — Dev Guide", encoding="utf-8")
    else:
        for path, text in _BOOKS.items():
            (repo_dir / path).parent.mkdir(parents=True, exist_ok=True)
            (repo_dir / path).write_text(text, encoding="utf-8", newline="\n")
    (repo_dir / "README.md").write_text('[![Version 5.5.0](https://img.shields.io/badge/version-5.5.0-green.svg)](VERSION)', encoding="utf-8")
    (repo_dir / "docs" / "CHECKLISTS.md").write_text("## Change Review Checklist\n| # | item |", encoding="utf-8")
    (drive_root / "state" / "state.json").write_text('{"spent_usd": 0}', encoding="utf-8")
    (drive_root / "memory" / "scratchpad.md").write_text("test scratchpad", encoding="utf-8")
    (drive_root / "memory" / "identity.md").write_text("I am Ouroboros.", encoding="utf-8")
    env = Env(repo_dir=repo_dir, drive_root=drive_root)
    memory = Memory(drive_root=drive_root, repo_dir=repo_dir)
    return env, memory


def foreign_tree(path: pathlib.Path) -> pathlib.Path:
    """A git tree whose only remote is someone else's repository: body_fact proves it foreign."""
    import subprocess

    path.mkdir(parents=True, exist_ok=True)
    if not (path / ".git").exists():
        for args in (("init", "-q"), ("remote", "add", "origin", "https://github.com/someone/elsewhere.git")):
            subprocess.run(["git", *args], cwd=path, check=True, capture_output=True)
    return path


def _build_system_text(task_overrides=None, *, context_mode="max", legacy=False):
    from ouroboros.context import build_llm_messages
    tmpdir = pathlib.Path(tempfile.mkdtemp())
    env, memory = _make_env_and_memory(tmpdir, legacy=legacy)
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


@pytest.mark.parametrize("mode", ["max", "low", "nano"])
def test_every_mode_carries_the_body_map_and_no_chapter_body(mode):
    _map_only(_build_system_text(context_mode=mode))
    # The task's own requirement adds the whole handbook in the same mode, the Map unchanged beside it.
    _required(_build_system_text(REQUIRED, context_mode=mode))


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


def test_forked_context_uses_canonical_global_cognition_not_child_noise(monkeypatch):
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
    (canonical / "memory" / "knowledge" / "overview.md").write_text(
        "CANONICAL_OVERVIEW", encoding="utf-8",
    )
    (child / "memory" / "knowledge" / "overview.md").write_text(
        "CHILD_OVERVIEW_NOISE", encoding="utf-8",
    )
    (canonical / "memory" / "deep_review.md").write_text(
        "CANONICAL_DEEP_REVIEW", encoding="utf-8",
    )
    (child / "memory" / "deep_review.md").write_text("CHILD_DEEP_NOISE", encoding="utf-8")
    env = Env(repo_dir=canonical_env.repo_dir, drive_root=child, budget_drive_root=canonical)
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    # A forked root names the canonical original under runtime_data (mode matrix: tests/test_authored_memory.py).
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "cyber_pro")
    ctx = ToolContext(repo_dir=env.repo_dir, drive_root=child, task_id="fork-cognition", is_direct_chat=True,
                      task_metadata={"budget_drive_root": str(canonical)})
    messages, _ = build_llm_messages(
        env=env,
        memory=Memory(child, repo_dir=env.repo_dir),
        task={"id": "fork-cognition", "text": "continue", "budget_drive_root": str(canonical)},
        ctx=ctx,
    )
    rendered = json.dumps(messages, ensure_ascii=False)

    assert "CANONICAL_OVERVIEW" in rendered
    assert "CHILD_OVERVIEW_NOISE" not in rendered
    # The report rides as a pointer to the CANONICAL source (its size, its address), never a
    # body, with an absolute canonical read address; actual dispatch of that call opens it.
    assert f"{len('CANONICAL_DEEP_REVIEW')} chars" in rendered
    assert "CANONICAL_DEEP_REVIEW" not in rendered and "CHILD_DEEP_NOISE" not in rendered
    call = {"root": "runtime_data", "path": str((canonical / "memory" / "deep_review.md").resolve())}
    pointer = f"Read: read_file(root='runtime_data', path={call['path']!r})"  # as the product spells it (repr)
    assert json.dumps(pointer, ensure_ascii=False)[1:-1] in rendered
    registry = ToolRegistry(repo_dir=env.repo_dir, drive_root=child)
    registry.set_context(ctx)
    read = registry.execute("read_file", call)
    assert "CANONICAL_DEEP_REVIEW" in read and "CHILD_DEEP_NOISE" not in read


def test_deep_review_is_a_historical_pointer_dated_only_by_its_header(tmp_path):
    import json

    from ouroboros.context import build_llm_messages
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    env, memory = _make_env_and_memory(tmp_path)
    path = env.drive_root / "memory" / "deep_review.md"
    ctx = ToolContext(repo_dir=env.repo_dir, drive_root=env.drive_root, task_id="history", is_direct_chat=True)
    registry = ToolRegistry(repo_dir=env.repo_dir, drive_root=env.drive_root)
    registry.set_context(ctx)
    for prefix, provenance in (
            ("LEGACY REPORT\n", "generated_at=unknown, source_revision=unknown (no provenance header)"),
            ("<!-- deep-review provenance: generated_at=2026-01-01T00:00:00Z, source_revision=unknown -->\n",
             "generated_at=2026-01-01T00:00:00Z, source_revision=unknown (its provenance header)")):
        original = prefix + "retained historical reasoning " * 400
        path.write_bytes(original.encode("utf-8"))  # pin the source bytes, not the host's text-mode EOL
        # A recent file mtime is not a review date, even on a copied old report.
        os.utime(path, (1900000000, 1900000000))
        messages, _ = build_llm_messages(
            env=env, memory=memory, task={"id": "history", "type": "task", "text": "continue"}, ctx=ctx,
        )
        rendered = json.dumps(messages, ensure_ascii=False)
        assert provenance in rendered and f"{len(original)} chars" in rendered
        assert "retained historical reasoning" not in rendered  # the report stays at its source
        assert "not a verdict on the current tree" in rendered
        assert "Read: read_file(root='runtime_data', path='memory/deep_review.md')" in rendered
        assert "retained historical reasoning" in registry.execute(
            "read_file", {"root": "runtime_data", "path": "memory/deep_review.md"})  # the call shown opens it
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
    seed_result(env.drive_root, task_id, {
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
    })

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
    seed_result(env.drive_root, predecessor_id, {
        "_schema_version": 1,
        "task_id": predecessor_id,
        "status": "completed",
        "objective": "o" * 700 + tail,
        "task_contract": {
            "objective": "o" * 700 + tail,
            "context": "never use native API",
            "delegation_budget": {"intent_note": "L1 asks L2 to spawn L3"},
        },
    })
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

    # The first request carries the brief and its reader; the contract stays at its source (THESIS I-7, C5).
    assert "predecessor_authority" in rendered and predecessor_id in rendered and "include_authority" in rendered
    assert tail not in rendered and "L1 asks L2 to spawn L3" not in rendered
    assert actor_read["status"] == "available"
    assert actor_read["authority"]["task_contract"]["objective"].endswith(tail)
    assert actor_read["authority"]["task_contract"]["delegation_budget"]["intent_note"] == "L1 asks L2 to spawn L3"


def test_unreadable_current_plan_review_source_returns_typed_startup_refusal(monkeypatch):
    from ouroboros.agent_startup_checks import validate_task_authority_sources

    tmpdir = pathlib.Path(tempfile.mkdtemp())
    env, _memory = _make_env_and_memory(tmpdir)
    seed_result(env.drive_root, "plan-broken", {"task_id": "plan-broken", "status": "running"})
    unreadable_result(monkeypatch, "plan-broken")

    refusal = validate_task_authority_sources(
        env, {"id": "plan-broken", "title": "Cat plan", "text": "continue"},
    )

    assert refusal["reason_code"] == "authority_source_unavailable"
    assert refusal["source"]["kind"] == "plan_review_state"
    assert refusal["human_label"] == "Cat plan"


def test_named_owner_source_resolves_beyond_automatic_recent_generations():
    from ouroboros.agent_startup_checks import validate_task_authority_sources
    from ouroboros.project_dialogue import build_owner_message_ref

    from tests import _dialogue as dialogue_test

    tmpdir = pathlib.Path(tempfile.mkdtemp())
    env, _memory = _make_env_and_memory(tmpdir)
    text = "old but exact owner authority"
    ref = build_owner_message_ref(
        chat_id=1, client_message_id="old-owner", ts="2026-08-01T00:00:00Z", text=text,
    )
    # The owner's row is in the oldest of five archived generations of the legacy chain, imported as the upgrade does.
    dialogue_test.seed(env.drive_root, archives={
        "chat_0001.jsonl": [{**ref, "direction": "in", "text": text}],
        **{f"chat_000{index}.jsonl": [{"direction": "in", "text": f"newer-{index}"}] for index in range(2, 6)}},
        live={"chat": [{"direction": "in", "text": "live"}]})
    dialogue_test.import_all(env.drive_root)
    task = {"id": "legacy-source", "origin_message_ref": ref}

    assert validate_task_authority_sources(env, task) == {}
    assert task["origin_message_text"] == text


@pytest.mark.parametrize("mode", ["max", "low"])
def test_an_external_surface_and_a_self_body_flag_carry_the_map_and_no_handbook(mode):
    """An external CLI task on another folder carries the same Body Map as any task; declaring the work
    Ouroboros's own (``context_requires_self_body_docs``) preloads no manual either. Only the contract's own
    requirement brings the whole handbook, on the same surface."""
    from ouroboros.contracts.task_contract import build_task_contract

    surface = {"workspace_root": "/tmp/example-workspace", "workspace_mode": "external", "actor_id": "cli",
               "metadata": {"source": "cli"}}
    _map_only(_build_system_text(surface, context_mode=mode))
    _map_only(_build_system_text({**surface, "context_requires_self_body_docs": True}, context_mode=mode))
    _required(_build_system_text({**surface, "context_requires_self_body_docs": True, **REQUIRED}, context_mode=mode))

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
    development_body = "# Review\n\nHow a change is reviewed.\n\n## Rules\n\nRead every source before committing.\n"
    (env.repo_dir / "docs" / "development" / "01-review.md").write_text(development_body, encoding="utf-8", newline="\n")
    memory = Memory(host.root, repo_dir=env.repo_dir)
    workspace = foreign_tree(tmp_path / "working-copy")  # proven foreign: the flag selects DESIGN, never the handbook
    project = create_project(host.root, "code-copy", name="Code Copy")
    entry = next(tool for tool in get_tools() if tool.name == "start_task")
    field = "context_requires_self_body_docs"
    parameters = entry.schema["parameters"]
    assert parameters["properties"][field]["type"] == "boolean"
    assert field not in parameters["required"]
    description = parameters["properties"][field]["description"]
    assert "Ouroboros code" in description and "copies elsewhere" in description
    explicit = parameters["properties"]["context_requires_development"]  # absent is not false (6A)
    assert explicit["type"] == "boolean" and "default" not in explicit and "context_requires_development" not in parameters["required"]
    events = []

    def emit(event):
        events.append(copy.deepcopy(event))
        _handle_promote_chat_to_task(event, host.ctx)

    tool_ctx = SimpleNamespace(
        repo_dir=env.repo_dir, system_repo_dir=env.repo_dir, drive_root=host.root, task_id="promoter",
        current_chat_id=1, task_metadata={}, pending_events=[], event_queue=SimpleNamespace(put_nowait=emit),
    )
    first = None
    # A later task in the SAME project must not inherit the first task's choice; the last one also requires the handbook.
    for flag, required in ((True, False), (None, False), (False, False), (True, True)):
        kwargs = ({} if flag is None else {field: flag}) | ({"context_requires_development": True} if required else {})
        result = entry.handler(tool_ctx, "Inspect this working copy", title="Working copy inspection",
                               project_id=project["id"],
                               workspace_root=str(workspace), **kwargs)
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
        # One governance block for every task: the Map names the handbook's chapters, the flag preloads none.
        assert content[0] == first and "- Review: `docs/development/01-review.md`" in content[0]["text"]
        assert "## Body map" in content[0]["text"] and "Read every source before committing." not in content[0]["text"]
        handbook = [block for block in content if block["text"].startswith("## DEVELOPMENT.md\n")]
        assert len(handbook) == int(required)
        if handbook:
            assert task["task_contract"]["context_requires_development"] is True
            assert content[1] == handbook[0] and "cache_control" in handbook[0]
            assert development_body in handbook[0]["text"]
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
        assert "## Body map" in child_content[0]["text"] and "- Review: `docs/development/01-review.md`" in child_content[0]["text"]
        assert "context_requires_development" not in child["task_contract"]  # no descendant inherits the requirement
        assert not any(block["text"].startswith("## DEVELOPMENT.md\n") for block in child_content)
    assert len(events) == len(host.pending) == 4 and not host.attempts


def test_evolution_and_deep_self_review_carry_the_body_map_without_a_handbook():
    """Owner 25.11.2026 (BIBLE P1): self-body work reads its chapters on demand from the Map's addresses; the task
    type preloads no manual in any mode (it used to keep both books whole in Max)."""
    for kind in ("evolution", "deep_self_review"):
        for mode in ("max", "low"):
            _map_only(_build_system_text({"type": kind}, context_mode=mode))
    _required(_build_system_text({"type": "evolution", **REQUIRED}, context_mode="max"))


def test_no_binding_or_room_preloads_the_handbook_only_the_contract_does(tmp_path):
    """Neither the active repo binding nor project membership decides the handbook any more (it used to key on the
    binding, D-DEV 2026-08-08): a proven-foreign folder, an unknown folder, a direct chat in a project room, an
    opted-out room, evolution in a room and an explicit self-body flag all carry the same Map alone. The task's own
    contract requirement brings it whole on the same binding; an explicit false keeps it out."""
    foreign = str(foreign_tree(tmp_path / "proj-tree"))
    cases = (
        {"project_id": "proj_sub", "workspace_root": foreign, "workspace_mode": "external"},
        {"project_id": "proj_sub", "workspace_root": "/tmp/proj-tree-unknown", "workspace_mode": "external"},
        {"project_id": "proj_sub", "_is_direct_chat": True},
        {"project_id": "proj_sub", "workspace": "none"},
        {"project_id": "proj_sub", "type": "evolution"},
        {"project_id": "proj_sub", "workspace_root": "/tmp/proj-tree", "workspace_mode": "external",
         "context_requires_self_body_docs": True},
        {"project_id": "proj_sub", "task_contract": {"context_requires_development": False}},
    )
    for task in cases:
        _map_only(_build_system_text(task, context_mode="max"))
    _map_only(_build_system_text({"project_id": "proj_sub", "_is_direct_chat": True}, context_mode="low"))
    _required(_build_system_text({"project_id": "proj_sub", "workspace_root": foreign, "workspace_mode": "external",
                                  **REQUIRED}, context_mode="max"))


def test_readme_and_checklists_are_never_inlined_and_the_atlas_names_the_checklists():
    for mode in ("max", "low"):
        text = _build_system_text(context_mode=mode)
        # The Map's Reference Atlas names the checklists (visible, never silently dropped); neither is inlined.
        assert "`docs/CHECKLISTS.md` — the questions my reviewers answer." in text
        assert "| # | item |" not in text and "img.shields.io" not in text


def test_without_a_renderable_book_the_on_demand_pointer_names_every_reference_doc(tmp_path):
    from ouroboros.context import build_llm_messages

    env, memory = _make_env_and_memory(tmp_path)
    (env.repo_dir / "docs" / "ARCHITECTURE.md").unlink()
    messages, _ = build_llm_messages(env=env, memory=memory, task={"id": "t", "type": "task", "text": "hello"})
    text = messages[0]["content"][0]["text"]
    assert "## Body map" not in text and "Reference book source unavailable: docs/ARCHITECTURE.md." in text
    assert "Reference docs available on demand" in text
    for name in ("docs/ARCHITECTURE.md", "docs/DEVELOPMENT.md", "README.md", "docs/CHECKLISTS.md"):
        assert f"`{name}`" in text


@pytest.mark.parametrize("mode", ["max", "low"])
def test_a_single_file_architecture_revision_rides_as_its_source_navigation(mode):
    """A legacy single-file book has no Map form: its source navigation (headings with physical line ranges)
    stands in every mode, never its body."""
    text = _build_system_text(context_mode=mode, legacy=True)
    assert "# Architecture (source navigation)\nFull source: `docs/ARCHITECTURE.md`" in text
    # Every heading with its physical line range: a parent's range keeps its complete subtree.
    assert "- Section A: `docs/ARCHITECTURE.md` lines 3-12" in text
    assert "- Section A child: `docs/ARCHITECTURE.md` lines 7-12" in text
    assert "- Section A detail: `docs/ARCHITECTURE.md` lines 9-12" in text
    assert "- Section B: `docs/ARCHITECTURE.md` lines 13-15" in text
    assert _ARCH_BODY_SENTINEL not in text  # full body NOT inlined
    assert "## Body map" not in text and HANDBOOK not in text


def test_low_mode_lists_the_handbook_chapters_for_task_and_direct_chat():
    for task in ({"type": "task"}, {"_is_direct_chat": True},
                 {"_is_direct_chat": True, "task_contract": {"context_requires_development": False}}):
        _map_only(_build_system_text(task, context_mode="low"))
    _required(_build_system_text({"_is_direct_chat": True, **REQUIRED}, context_mode="low"))


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
