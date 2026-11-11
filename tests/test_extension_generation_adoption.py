"""The server→worker extension generation channel (W3B-F1).

Task workers load extensions ONCE, at spawn. A skill enabled after that was
invisible to every task the process went on to serve until the pool respawned:
the model calling the fresh surface got "Unknown tool" while ``/api/extensions``
truthfully reported ``live_loaded``. These pins cover the durable carrier
(``state/extension_generation.json``), both natural points a worker notices with
(task start and the dispatch miss), the symmetric disable direction, and the
three properties that keep the mechanism from being worse than the defect:
write-if-changed publication (the publish/adopt cycle terminates), one reload
per DISTINCT published generation (a worker that structurally cannot converge
degrades to the old behaviour instead of reloading before every task), and
fail-closed on a registry it cannot read.
"""

from __future__ import annotations

import os
import pathlib
import types

import pytest

from ouroboros import extension_loader
from ouroboros.extension_companion import init_server_process_pid
from ouroboros.extension_reconcile_queue import (
    adopt_published_extension_generation,
    extension_generation_path,
    publish_extension_generation,
    published_extension_generation,
)
from ouroboros.extension_registry_state import live_extension_fingerprint
from ouroboros.skill_loader import save_enabled
from ouroboros.utils import atomic_write_json

from tests._shared import clean_extension_runtime_state
from tests._extension_loader_shared import _prepare_extension
from tests._extension_loader_shared import (  # noqa: F401  (autouse fixture applies on import)
    _clear_loader_state,
)

_PROBE_PLUGIN = (
    "def register(api):\n"
    "    api.register_tool('echo', lambda ctx, message='hi': f'echo: {message}',\n"
    "                      description='echo probe', schema={})\n"
)
# Imports cleanly in the publisher and raises in the adopter — the shape that
# would let a worker's local failure demote a globally healthy skill.
_BROKEN_PLUGIN = "raise RuntimeError('this extension never imports')\n"


@pytest.fixture(autouse=True)
def _restore_server_pid():
    original = os.environ.get("OUROBOROS_SERVER_PROCESS_PID")
    yield
    init_server_process_pid(int(original) if original else -1)
    if original is None:
        os.environ.pop("OUROBOROS_SERVER_PROCESS_PID", None)


def _as_server() -> None:
    init_server_process_pid(os.getpid())


def _as_worker() -> None:
    init_server_process_pid(999999)


def _published_by_a_server_that_loaded_it(tmp_path: pathlib.Path, name: str = "genprobe"):
    """Drive the SERVER side for real: publish empty, then load and publish again.

    Both generation values are genuine ``publish_extension_generation`` outputs
    of this tree, so a test that later hands one of them to the worker side is
    replaying a real publication rather than a value it invented.
    """
    _as_server()
    loaded, repo_root, drive_root = _prepare_extension(
        tmp_path, name, _PROBE_PLUGIN, permissions=["tool"])
    empty_generation = publish_extension_generation(drive_root)
    results = extension_loader.reload_all(drive_root, lambda: {}, repo_path=str(repo_root))
    assert results == {name: None}, results  # None == live, no load error
    loaded_generation = published_extension_generation(drive_root)
    assert loaded_generation and loaded_generation != empty_generation
    surface = extension_loader.extension_surface_name(name, "echo")
    assert extension_loader.get_tool(surface) is not None
    return loaded, repo_root, drive_root, surface, empty_generation, loaded_generation


def _settings_and_repo(monkeypatch, repo_root: pathlib.Path) -> None:
    from ouroboros import config

    monkeypatch.setattr(config, "load_settings", lambda: {})
    monkeypatch.setattr(config, "get_skills_repo_path", lambda: str(repo_root))


# ---------------------------------------------------------------------------
# The defect itself, at both natural points.
# ---------------------------------------------------------------------------


def test_worker_with_a_stale_generation_loads_a_skill_enabled_after_its_spawn(
        tmp_path, monkeypatch):
    """THE defect pin: the task-start point, driven through the worker's own seam."""
    from supervisor import worker_process

    _loaded, repo_root, drive_root, surface, _empty, published = (
        _published_by_a_server_that_loaded_it(tmp_path))
    _settings_and_repo(monkeypatch, repo_root)

    # The worker spawned BEFORE the enable: nothing live, the published
    # generation already names a set it has never seen.
    clean_extension_runtime_state()
    _as_worker()
    assert extension_loader.get_tool(surface) is None, (
        "the worker's pre-enable registry is the premise of this pin")

    worker_process._adopt_published_extensions(str(drive_root))

    assert extension_loader.get_tool(surface) is not None, (
        "a skill enabled after the worker spawned is still invisible to its tasks")
    assert live_extension_fingerprint() == published


def test_dispatching_an_unknown_extension_surface_adopts_before_answering(
        tmp_path, monkeypatch):
    """The dispatch-miss point: an enable that lands MID-task is still dispatchable."""
    from ouroboros.tools.extension_dispatch import _extension_dispatch_candidate

    _loaded, repo_root, drive_root, surface, _empty, _published = (
        _published_by_a_server_that_loaded_it(tmp_path))
    _settings_and_repo(monkeypatch, repo_root)
    clean_extension_runtime_state()
    _as_worker()

    ctx = types.SimpleNamespace(
        task_metadata={}, budget_drive_root=str(drive_root), drive_root=str(drive_root))
    descriptor, host_attested_unavailable = _extension_dispatch_candidate(ctx, surface)

    assert descriptor is not None, "the dispatch miss answered Unknown tool without probing"
    assert host_attested_unavailable is False
    assert descriptor.get("skill") == "genprobe"


_SLOW_PLUGIN = _PROBE_PLUGIN.replace("schema={})", "schema={}, timeout_sec=1200)")


def _update_published_by_another_server(repo_root, drive_root, name="genprobe"):
    """Review a new payload and let a genuinely separate server process publish it."""
    from ouroboros.skill_loader import SkillReviewState, find_skill, save_review_state
    from tests.test_extension_freshness_cross_process import _in_another_process

    (repo_root / name / "plugin.py").write_text(_SLOW_PLUGIN, encoding="utf-8")
    updated = find_skill(drive_root, name, repo_path=str(repo_root))
    save_review_state(drive_root, name, SkillReviewState(status="pass", content_hash=updated.content_hash))
    before = published_extension_generation(drive_root)
    _in_another_process(
        "import os, pathlib\nfrom ouroboros.extension_companion import init_server_process_pid\n"
        "init_server_process_pid(os.getpid())\nfrom ouroboros import extension_loader\n"
        "from ouroboros.extension_reconcile_queue import publish_extension_generation\n"
        "extension_loader.reload_all(pathlib.Path(drive), lambda: {}, repo_path=repo)\n"
        "publish_extension_generation(pathlib.Path(drive))\n",
        drive=str(drive_root), repo=str(repo_root))
    published = published_extension_generation(drive_root)
    assert published and published != before, "the separate server did not publish the update"
    return published


def _live_worker_behind_an_update(tmp_path, monkeypatch):
    """A worker that loaded ``genprobe`` (60 s) before a reviewed update (1200 s) was published."""
    from ouroboros.tools.registry import ToolRegistry

    _loaded, repo_root, drive_root, surface, _empty, _published = (
        _published_by_a_server_that_loaded_it(tmp_path))
    _settings_and_repo(monkeypatch, repo_root)
    _as_worker()
    assert extension_loader.get_tool(surface)["timeout_sec"] == 60
    _update_published_by_another_server(repo_root, drive_root)
    assert extension_loader.get_tool(surface)["timeout_sec"] == 60, "premise: the worker still holds the old descriptor"
    reloads: list[str] = []
    real_reload = extension_loader.reload_all
    monkeypatch.setattr(extension_loader, "reload_all",
                        lambda *a, **kw: (reloads.append("reload"), real_reload(*a, **kw))[1])
    registry = ToolRegistry(repo_dir=pathlib.Path(__file__).resolve().parents[1], drive_root=drive_root)
    return registry, drive_root, surface, reloads


def _batch(registry, drive_root, *calls):
    from ouroboros.loop_tool_execution import execute_tool_batch

    logs = drive_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    results, control = execute_tool_batch(list(calls), registry, logs, "genprobe-task", None, {})
    assert control is None
    starts = {}
    for line in (logs / "tools.jsonl").read_text(encoding="utf-8").splitlines():
        row = __import__("json").loads(line)
        if row.get("type") == "tool_call_started":
            starts[row["tool_call_id"]] = row
    return results, starts


def _call(call_id, surface, message="hi"):
    import json
    return {"id": call_id, "function": {"name": surface, "arguments": json.dumps({"message": message})}}


def test_an_update_published_during_a_live_worker_dispatches_under_the_new_timeout_once(tmp_path, monkeypatch):
    """#2933 through the real loop: the outer timeout and the dispatch follow ONE adopted generation."""
    import ouroboros.safety

    monkeypatch.setattr(ouroboros.safety, "check_safety", lambda *a, **kw: (True, ""))
    registry, drive_root, surface, reloads = _live_worker_behind_an_update(tmp_path, monkeypatch)

    results, starts = _batch(registry, drive_root, _call("c1", surface))

    assert [row["is_error"] for row in results] == [False], results
    assert results[0]["result"] == "echo: hi"
    assert starts["c1"]["timeout_sec"] >= 1203, "the outer clock was chosen from the pre-update descriptor"
    assert reloads == ["reload"], "one published generation costs exactly one reload"
    # The adopted generation now serves named discovery, the catalog and the next call without reloading.
    assert registry.get_schema_by_name(surface) is not None
    assert surface in {row["function"]["name"] for row in registry.schemas()}
    results, starts = _batch(registry, drive_root, _call("c2", surface, "again"))
    assert results[0]["result"] == "echo: again" and starts["c2"]["timeout_sec"] >= 1203
    assert reloads == ["reload"]


def test_named_discovery_adopts_the_update_before_answering(tmp_path, monkeypatch):
    """enable_tools/get_schema_by_name never answers Not found for a surface the update kept."""
    registry, _drive_root, surface, reloads = _live_worker_behind_an_update(tmp_path, monkeypatch)

    schema = registry.get_schema_by_name(surface)

    assert schema is not None and schema["function"]["name"] == surface
    assert reloads == ["reload"]


def test_a_revocation_after_the_timeout_was_chosen_refuses_before_the_handler(tmp_path, monkeypatch):
    """Negative control: the bound descriptor never skips the physical dispatch's fresh liveness read."""
    import ouroboros.safety
    from ouroboros import loop_tool_execution

    monkeypatch.setattr(ouroboros.safety, "check_safety", lambda *a, **kw: (True, ""))
    registry, drive_root, surface, _reloads = _live_worker_behind_an_update(tmp_path, monkeypatch)
    real_timeout = loop_tool_execution._get_tool_timeout

    def choose_then_revoke(tools, name, args=None):
        chosen = real_timeout(tools, name, args)
        save_enabled(drive_root, "genprobe", False)  # the owner disables between the clock and the dispatch
        return chosen

    monkeypatch.setattr(loop_tool_execution, "_get_tool_timeout", choose_then_revoke)
    results, starts = _batch(registry, drive_root, _call("c1", surface))

    assert starts["c1"]["timeout_sec"] >= 1203
    assert results[0]["is_error"] is True
    assert results[0]["result_meta"].get("tool_code") == "EXTENSION_UNAVAILABLE" or "not allowed to dispatch" in results[0]["result"]
    assert "echo:" not in results[0]["result"], "the handler ran after a revocation"


def test_parallel_same_name_calls_each_dispatch_the_descriptor_their_own_timeout_chose(tmp_path, monkeypatch):
    """Negative control for a binding shared by tool name: both calls resolve before either dispatches."""
    import threading

    from ouroboros import loop_tool_execution
    from ouroboros.tools import extension_dispatch
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tools.tool_result import ToolResult

    surface = extension_loader.extension_surface_name("genprobe", "echo")
    resolved, dispatched = [], []
    both_resolved = threading.Barrier(2, timeout=10)
    lock = threading.Lock()

    def candidate(_ctx, name):
        with lock:
            resolved.append(name)
            descriptor = {"name": name, "skill": "genprobe", "timeout_sec": 60 * len(resolved), "handler": print}
        both_resolved.wait()
        return descriptor, False

    def dispatch(_ctx, name, ext_tool, args, **_kwargs):
        dispatched.append((args["message"], ext_tool["timeout_sec"]))
        return ToolResult(status="ok", code="OK", text=f"{args['message']}:{ext_tool['timeout_sec']}")

    monkeypatch.setattr(extension_dispatch, "_extension_dispatch_candidate", candidate)
    monkeypatch.setattr(extension_dispatch, "_dispatch_extension_tool_result", dispatch)
    monkeypatch.setattr(loop_tool_execution, "tool_calls_can_run_parallel", lambda calls: True)
    monkeypatch.setattr(loop_tool_execution, "load_settings", lambda: {})  # no global floor hides the per-call timeout
    monkeypatch.delenv("OUROBOROS_TOOL_TIMEOUT_SEC", raising=False)
    drive_root = tmp_path / "drive"
    registry = ToolRegistry(repo_dir=pathlib.Path(__file__).resolve().parents[1], drive_root=drive_root)

    results, starts = _batch(registry, drive_root, _call("a", surface, "a"), _call("b", surface, "b"))

    assert len(resolved) == 2, "a call resolved again instead of consuming its own binding"
    for row in results:
        message, timeout = row["result"].split(":")
        assert starts[row["tool_call_id"]]["timeout_sec"] == int(timeout) + 3, (row, starts)
    assert sorted(timeout for _message, timeout in dispatched) == [60, 120]


def _publish_replacement(loaded, repo_root, drive_root, surface, plugin, *, effect, replacement):
    """A reviewed replacement declaring ``replacement`` publishes through a real reload; its handler answers ``new: ...``."""
    from ouroboros.skill_loader import SkillReviewState, find_skill, save_review_state

    before = extension_loader.get_tool(surface)["extension_generation"]
    manifest = loaded.skill_dir / "SKILL.md"
    manifest.write_text(manifest.read_text().replace(f"effect: {effect}", f"effect: {replacement}"))
    (loaded.skill_dir / "plugin.py").write_text(plugin.replace("f'", "f'new: ", 1))
    updated = find_skill(drive_root, loaded.name, repo_path=str(repo_root))
    save_review_state(drive_root, loaded.name, SkillReviewState(status="pass", content_hash=updated.content_hash))
    assert extension_loader.reload_all(drive_root, lambda: {}, repo_path=str(repo_root)) == {loaded.name: None}
    assert extension_loader.get_tool(surface)["extension_generation"] != before


@pytest.mark.parametrize("effect,replacement,reload,blocked", [
    ("write", "read", True, True),  # a replacement's read never lends its skip to the bound write handler
    ("write", "write", True, True),
    ("read", "read", True, False),  # the bound handler's own reviewed read skips, exactly as without the reload
    ("read", "write", True, False),  # nor does a replacement's write become the bound read handler's declaration
    ("write", "write", False, True),
    ("read", "read", False, False),
])
def test_safety_assesses_the_timeout_bound_handler_under_its_own_reviewed_declaration(
        tmp_path, monkeypatch, effect, replacement, reload, blocked):
    """A real loader swap after timeout selection: the handler that runs is the bound one, judged by its own publication."""
    from ouroboros import safety
    from ouroboros.tools.registry import ToolRegistry
    from tests._safety_fixtures import reset_safety_stores

    _as_server()
    loaded, repo_root, drive_root = _prepare_extension(
        tmp_path, "genprobe", _PROBE_PLUGIN, permissions=["tool"],
        extra_frontmatter=f"tools:\n  - name: echo\n    effect: {effect}\n")
    _settings_and_repo(monkeypatch, repo_root)
    assert extension_loader.reload_all(drive_root, lambda: {}, repo_path=str(repo_root)) == {"genprobe": None}
    surface = extension_loader.extension_surface_name("genprobe", "echo")
    registry = ToolRegistry(repo_dir=pathlib.Path(__file__).resolve().parents[1], drive_root=drive_root)
    assessments = []
    monkeypatch.setattr(safety, "get_safety_mode", lambda *a: "full")
    monkeypatch.setattr(safety, "_run_llm_check", lambda *a, **kw: (assessments.append(a[0]), (False, "inert refusal"))[1])
    monkeypatch.setattr(safety, "_emit_durable_safety_event", lambda *a, **kw: None)
    reset_safety_stores()
    real_timeout = registry.get_timeout

    def select_then_reload(name):
        timeout = real_timeout(name)
        if reload:
            _publish_replacement(loaded, repo_root, drive_root, surface, _PROBE_PLUGIN,
                                 effect=effect, replacement=replacement)
        return timeout

    monkeypatch.setattr(registry, "get_timeout", select_then_reload)
    results, _starts = _batch(registry, drive_root, _call("bound-effect", surface))
    assert results[0]["is_error"] is blocked, results
    assert assessments == ([surface] if blocked else []), assessments
    if blocked:
        assert "inert refusal" in results[0]["result"]
    else:
        assert results[0]["result"] == "echo: hi", "not the timeout-bound handler"


# One captured generation per invocation: the bound handler's own reviewed declaration and routing reach the place
# guard and the Safety brief; a replacement lends it nothing; an independent revocation still refuses.
_SPEAKER_PLUGIN = (
    "def register(api):\n"
    "    api.register_tool('say', lambda ctx, channel='', text='': f'said {text} in {channel}',\n"
    "                      description='speaker probe', schema={})\n"
)
_SPEAKS = "tools:\n  - name: say\n    effect: message\n    target: channel\n    conversation: channel\n"


def _supervisor(monkeypatch, verdict="SAFE", runtime="advanced"):
    """The real Safety path; only the physical model call is recorded and answers ``verdict`` (an exception: no answer)."""
    import json

    from ouroboros import safety
    from tests._safety_fixtures import reset_safety_stores

    prompts: list[str] = []
    reset_safety_stores()
    monkeypatch.setattr(safety, "_SAFETY_STORM_UNTIL", 0)
    monkeypatch.setattr(safety, "_CHAIN_STORMS", {})
    monkeypatch.setattr(safety, "_resolve_safety_routing", lambda: (False, False, None))
    monkeypatch.setattr(safety, "get_safety_model", lambda: "fixture/checker")
    monkeypatch.setattr(safety, "get_safety_mode", lambda *_a: "full")
    monkeypatch.setattr(safety, "get_runtime_mode", lambda: runtime)
    monkeypatch.setattr(safety, "_emit_durable_safety_event", lambda *_a, **_k: None)

    def model(**kwargs):
        prompts.append(kwargs["user_prompt"])
        if isinstance(verdict, Exception):
            raise verdict
        return {"content": json.dumps({"status": verdict, "reason": "fixture verdict"})}, {}

    monkeypatch.setattr(safety, "_safety_model_call", model)
    return prompts


def _brief_place(prompt):
    """The ``subject.place`` host facts of one Supervisor prompt."""
    import json

    evidence = prompt.split("Host evidence (provenance is not consent):\n", 1)[1].split("\nProposed tool call:\n", 1)[0]
    return json.loads(evidence)["subject"]["place"]


def _speaker_call(tmp_path, monkeypatch, *, replacement="message", then=None, out_of_process=False):
    """Bind a real message connector's call to one Active place, then before it dispatches publish a reviewed
    replacement declaring ``replacement`` (None: no replacement) and run ``then(drive_root, place)``."""
    import json

    from ouroboros.places import connect_place
    from ouroboros.tools import extension_dispatch
    from ouroboros.tools.registry import ToolRegistry

    _as_server()
    loaded, repo_root, drive_root = _prepare_extension(
        tmp_path, "speaker", _SPEAKER_PLUGIN, permissions=["tool"], extra_frontmatter=_SPEAKS)
    _settings_and_repo(monkeypatch, repo_root)
    assert extension_loader.reload_all(drive_root, lambda: {}, repo_path=str(repo_root)) == {"speaker": None}
    place = connect_place(drive_root, connector="speaker", provider="chat", account_id="acct",
                          conversation_id="C2", scope="conversation", label="Room two")
    surface = extension_loader.extension_surface_name("speaker", "say")
    if out_of_process:  # the descriptor as an isolated payload's catalog publishes it: its child loads the payload
        real_candidate = extension_dispatch._extension_dispatch_candidate
        monkeypatch.setattr(extension_dispatch, "_extension_dispatch_candidate", lambda ctx, name: (
            lambda found: ({**found[0], "out_of_process": True} if found[0] else None, found[1]))(real_candidate(ctx, name)))
    registry = ToolRegistry(repo_dir=pathlib.Path(__file__).resolve().parents[1], drive_root=drive_root)
    real_timeout = registry.get_timeout

    def bind_then_replace(name):
        timeout = real_timeout(name)
        if replacement:
            _publish_replacement(loaded, repo_root, drive_root, surface, _SPEAKER_PLUGIN,
                                 effect="message", replacement=replacement)
        if then:
            then(drive_root, place)
        return timeout

    monkeypatch.setattr(registry, "get_timeout", bind_then_replace)
    results, _starts = _batch(registry, drive_root, {"id": "say-1", "function": {
        "name": surface, "arguments": json.dumps({"channel": "C2", "text": "hi"})}})
    return results[0], place


@pytest.mark.parametrize("runtime,verdict,refusal", [  # a flag at either level returns words unsent (the final assessment)
    ("advanced", "SUSPICIOUS", "⚠️ SAFETY_RETURNED: not run;"),
    ("advanced", "DANGEROUS", "⚠️ SAFETY_RETURNED: not run;"),
    ("cyber_pro", "DANGEROUS", "⚠️ SAFETY_RETURNED: not run;"),
    ("cyber_pro", RuntimeError("checker route down"), "⚠️ SAFETY_UNAVAILABLE: not run."),
])
@pytest.mark.parametrize("replacement", ["message", "read", None])
def test_a_bound_message_handler_keeps_its_speech_refusal_across_a_replacement(
        tmp_path, monkeypatch, runtime, verdict, refusal, replacement):
    """rl-20261107T231911: after any reviewed replacement the bound message handler is still speech; a first flag or
    an unanswered check refuses it in every runtime mode, the handler never runs and the brief names its own route."""
    prompts = _supervisor(monkeypatch, verdict, runtime)

    result, place = _speaker_call(tmp_path, monkeypatch, replacement=replacement)

    assert result["is_error"] is True and refusal in result["result"], result
    assert "said hi in C2" not in result["result"], "the bound message handler ran past a speech refusal"
    facts = _brief_place(prompts[0])
    assert (facts["effect"], facts["declared"], facts["targets"]) == ("message", True, {"channel": ["C2"]}), facts
    assert [row["place_id"] for row in facts["places"]] == [place.place_id]


@pytest.mark.parametrize("replacement", ["read", "write"])
def test_a_checked_bound_handler_runs_with_its_own_route_in_the_brief(tmp_path, monkeypatch, replacement):
    """The Safety brief describes the handler that runs, never the replacement's declaration read by name."""
    prompts = _supervisor(monkeypatch, "SAFE")

    result, place = _speaker_call(tmp_path, monkeypatch, replacement=replacement)

    assert result["is_error"] is False and result["result"] == "said hi in C2", result
    facts = _brief_place(prompts[0])
    assert (facts["effect"], facts["declared"], facts["targets"]) == ("message", True, {"channel": ["C2"]}), facts
    assert [row["place_id"] for row in facts["places"]] == [place.place_id]


def test_a_bound_message_handler_still_meets_its_conversations_author(tmp_path, monkeypatch):
    """One author per conversation holds for the bound message handler: the guard refuses before Safety is paid."""
    from ouroboros.conversation_lease import PresenceTurnGate
    from ouroboros.presence_bindings import conversation_key

    prompts, leases = _supervisor(monkeypatch, "SAFE"), []

    def another_task_authors(drive_root, _place):
        lease = PresenceTurnGate(state_root=drive_root / "state").acquire(conversation_key("chat", "acct", "C2"))
        lease.bind_author("other-author")
        leases.append(lease)

    try:
        result, _place = _speaker_call(tmp_path, monkeypatch, then=another_task_authors)
    finally:
        for lease in leases:
            lease.release()
    assert result["is_error"] is True and "tell_task(task_id='other-author'" in result["result"], result
    assert prompts == [], "Safety assessed a call the place guard refuses"


@pytest.mark.parametrize("revocation,refusal", [("disabled", "not allowed to dispatch"), ("paused", "Paused")])
def test_an_independent_revocation_after_binding_still_refuses(tmp_path, monkeypatch, revocation, refusal):
    """Negative control: the captured generation never stands in for fresh liveness or the place's current state."""
    from ouroboros.places import edit_place

    prompts = _supervisor(monkeypatch, "SAFE")

    def revoke(drive_root, place):
        if revocation == "disabled":
            save_enabled(drive_root, "speaker", False)
        else:
            edit_place(drive_root, place.place_id, "set_state", author="owner", state="paused")

    result, _place = _speaker_call(tmp_path, monkeypatch, then=revoke)

    assert result["is_error"] is True and refusal in result["result"], result
    assert "said" not in result["result"] and prompts == []


@pytest.mark.parametrize("replacement", ["read", None])
def test_an_out_of_process_call_never_runs_a_replacement_under_its_bound_declaration(
        tmp_path, monkeypatch, replacement):
    """An isolated payload's child loads the live payload: after a replacement it would run the new code under the
    bound declaration (here a read skip), so the call refuses before Safety; unreplaced, it reaches the child."""
    from ouroboros import extension_process_runner

    prompts, children = _supervisor(monkeypatch, "SAFE"), []
    monkeypatch.setattr(extension_process_runner, "dispatch_extension_tool_subprocess",
                        lambda ext_tool, _ctx, args: children.append(ext_tool["content_hash"]) or "child ran")

    result, _place = _speaker_call(tmp_path, monkeypatch, replacement=replacement, out_of_process=True)

    if replacement:
        assert result["is_error"] is True and "replaced" in result["result"], result
        assert children == [] and prompts == []
    else:
        # The child's own result, then the host's facts for the place it spoke into (another place than the focus).
        assert result["result"].split("\n\n[DESTINATION] ")[0] == "child ran", result
        assert len(children) == 1 and len(prompts) == 1, result


def _another_caller_adopts_first(monkeypatch):
    """Deterministic interleaving: another caller's real adoption lands after this call saw the stale descriptor
    and before its own adoption, which then finds nothing left to reload."""
    from ouroboros.tools import extension_dispatch

    real, actions = extension_dispatch._adopt_extension_generation, []

    def adopt(ctx):
        if not actions:
            actions.append(real(ctx)["action"])  # the other caller
        actions.append(real(ctx)["action"])
        return {"action": actions[-1]}

    monkeypatch.setattr(extension_dispatch, "_adopt_extension_generation", adopt)
    return actions


def _observations(monkeypatch):
    seen: list[str] = []
    real = extension_loader.is_extension_live
    monkeypatch.setattr(extension_loader, "is_extension_live",
                        lambda skill, *a, **kw: (seen.append(skill), real(skill, *a, **kw))[1])
    return seen


@pytest.mark.parametrize("entry", ["dispatch", "named", "catalog"])
def test_a_stale_descriptor_another_caller_adopted_first_is_served_from_the_live_generation(
        tmp_path, monkeypatch, entry):
    """This call's own adoption reads in_sync, yet the registry changed after its stale observation: the
    now-live descriptor is observed again and served (never hidden or refused), with the one reload paid."""
    import ouroboros.safety

    monkeypatch.setattr(ouroboros.safety, "check_safety", lambda *a, **kw: (True, ""))
    registry, drive_root, surface, reloads = _live_worker_behind_an_update(tmp_path, monkeypatch)
    actions = _another_caller_adopts_first(monkeypatch)
    if entry == "dispatch":
        results, starts = _batch(registry, drive_root, _call("c1", surface))
        assert results[0]["result"] == "echo: hi" and starts["c1"]["timeout_sec"] >= 1203, results
    elif entry == "named":
        assert (registry.get_schema_by_name(surface) or {}).get("function", {}).get("name") == surface
    else:
        assert surface in {row["function"]["name"] for row in registry.schemas()}
    assert actions[0] == "reloaded" and actions[1] != "reloaded", actions
    assert reloads == ["reload"]


@pytest.mark.parametrize("entry", ["dispatch", "catalog"])
def test_an_unstamped_descriptor_compares_the_generation_each_read_observed(tmp_path, monkeypatch, entry):
    """A legacy descriptor without its own stamp takes its bundle's digest inside the read's own lock hold, so a
    bundle another caller replaced between the stale read and this call's adoption is seen, never read twice."""
    from ouroboros import extension_registry_state as state
    from ouroboros.tools import extension_dispatch
    from ouroboros.tools.registry import ToolRegistry

    surface = extension_loader.extension_surface_name("legacy", "echo")
    monkeypatch.setitem(state._tools, surface, {"name": surface, "skill": "legacy", "description": "legacy", "schema": {}})
    monkeypatch.setitem(state._extensions, "legacy", state._ExtensionRegistrations(generation_digest="old"))
    monkeypatch.setattr(extension_loader, "is_extension_live",
                        lambda skill, *a, **kw: state._extensions[skill].generation_digest == "new")

    def another_caller_reloaded(_ctx):
        state._extensions["legacy"] = state._ExtensionRegistrations(generation_digest="new")
        return {"action": "in_sync"}

    monkeypatch.setattr(extension_dispatch, "_adopt_extension_generation", another_caller_reloaded)
    drive_root = tmp_path / "drive"
    if entry == "dispatch":
        ctx = types.SimpleNamespace(task_metadata={}, budget_drive_root=str(drive_root), drive_root=str(drive_root))
        descriptor, unavailable = extension_dispatch._extension_dispatch_candidate(ctx, surface)
        assert descriptor is not None and descriptor["extension_generation"] == "new" and not unavailable
    else:
        registry = ToolRegistry(repo_dir=pathlib.Path(__file__).resolve().parents[1], drive_root=drive_root)
        monkeypatch.setattr(registry, "_mcp_rows", lambda **kwargs: [])
        assert surface in {row["function"]["name"] for row in registry.schemas()}


@pytest.mark.parametrize("entry", ["dispatch", "catalog"])
def test_a_disabled_skill_with_no_new_generation_keeps_one_observation_and_its_refusal(
        tmp_path, monkeypatch, entry):
    """Unchanged control: the adoption finds the registry as observed, so nothing is observed or reloaded again."""
    from ouroboros.tools import extension_dispatch
    from ouroboros.tools.registry import ToolRegistry

    loaded, repo_root, drive_root, surface, _empty, _published = _published_by_a_server_that_loaded_it(tmp_path)
    _settings_and_repo(monkeypatch, repo_root)
    _as_worker()
    save_enabled(drive_root, loaded.name, False)
    reloads: list[str] = []
    monkeypatch.setattr(extension_loader, "reload_all", lambda *a, **kw: reloads.append("reload") or {})
    seen = _observations(monkeypatch)
    if entry == "dispatch":
        ctx = types.SimpleNamespace(task_metadata={}, budget_drive_root=str(drive_root), drive_root=str(drive_root))
        assert extension_dispatch._extension_dispatch_candidate(ctx, surface) == (None, True)
    else:
        registry = ToolRegistry(repo_dir=pathlib.Path(__file__).resolve().parents[1], drive_root=drive_root)
        assert surface not in {row["function"]["name"] for row in registry.schemas()}
    assert seen == [loaded.name] and reloads == [], (seen, reloads)


def test_a_worker_adopts_the_disable_direction_too(tmp_path, monkeypatch):
    """Symmetry: the surface LEAVES a worker that already had it live."""
    loaded, repo_root, drive_root, surface, empty_generation, _published = (
        _published_by_a_server_that_loaded_it(tmp_path))
    _settings_and_repo(monkeypatch, repo_root)

    # This process now holds exactly what a worker spawned before the disable
    # holds. The owner disables, and the server republishes — verbatim the
    # empty-set generation this same publisher minted above.
    _as_worker()
    save_enabled(drive_root, loaded.name, False)
    atomic_write_json(extension_generation_path(drive_root), {
        "schema_version": 1, "generation": empty_generation, "published_at": "2026-01-01T00:00:00Z",
    })

    result = adopt_published_extension_generation(drive_root, lambda: {}, repo_path=str(repo_root))

    assert result["action"] == "reloaded", result
    assert result["converged"] is True, result
    assert extension_loader.get_tool(surface) is None, "disabled surface survived the adopt"


# ---------------------------------------------------------------------------
# The three properties that keep the mechanism cheaper than the defect.
# ---------------------------------------------------------------------------


def test_publish_extension_generation_is_write_if_changed(tmp_path):
    """The property the publish/adopt cycle TERMINATES on.

    An adopting worker's ``reload_all`` sends a reconcile request per skill; the
    server reconciles and announces again. That loop closes only because
    reconciling an already-live extension leaves the live fingerprint alone and
    therefore publishes nothing new — otherwise every adopt would mint a
    generation for the next adopt to chase.
    """
    _loaded, _repo_root, drive_root, _surface, _empty, published = (
        _published_by_a_server_that_loaded_it(tmp_path))
    path = extension_generation_path(drive_root)
    before = path.read_bytes()
    stat_before = path.stat().st_mtime_ns

    assert publish_extension_generation(drive_root) == published
    assert publish_extension_generation(drive_root) == published

    assert path.read_bytes() == before, "an unchanged live set rewrote the marker"
    assert path.stat().st_mtime_ns == stat_before, "the marker was rewritten byte-identically"

    clean_extension_runtime_state()
    assert publish_extension_generation(drive_root) != published
    assert path.read_bytes() != before, "a CHANGED live set must publish"


def test_a_generation_a_worker_cannot_converge_on_costs_exactly_one_reload(
        tmp_path, monkeypatch):
    """Bounded: structural divergence degrades to the pre-fix behaviour.

    A skill that loads in the server and not here can never make the local
    fingerprint equal the published one. Without the per-generation guard the
    worker would spend a full ``reload_all`` before every single task.
    """
    _as_worker()
    loaded, repo_root, drive_root = _prepare_extension(
        tmp_path, "brokenprobe", _BROKEN_PLUGIN, permissions=["tool"])
    atomic_write_json(extension_generation_path(drive_root), {
        "schema_version": 1, "generation": "a" * 32, "published_at": "2026-01-01T00:00:00Z",
    })
    calls: list[str] = []
    real_reload = extension_loader.reload_all
    monkeypatch.setattr(
        extension_loader, "reload_all",
        lambda *a, **kw: (calls.append("reload"), real_reload(*a, **kw))[1])

    first = adopt_published_extension_generation(drive_root, lambda: {}, repo_path=str(repo_root))
    second = adopt_published_extension_generation(drive_root, lambda: {}, repo_path=str(repo_root))
    third = adopt_published_extension_generation(drive_root, lambda: {}, repo_path=str(repo_root))

    assert first["action"] == "reloaded" and first["converged"] is False, first
    assert second["action"] == "already_adopted", second
    assert third["action"] == "already_adopted", third
    assert calls == ["reload"], calls
    # ...and the worker's local load failure did NOT demote a globally enabled
    # skill: reload_all never asks for the enable-path revert.
    from ouroboros.skill_loader import load_enabled

    assert load_enabled(drive_root, loaded.name) is True, (
        "a worker-side adopt disabled a skill for the whole install")


def test_a_reload_that_raises_leaves_a_typed_failure_fact_and_still_costs_one_reload(tmp_path, monkeypatch):
    """A reload that RAISES keeps the exactly-one-reload contract (the generation
    stays marked, later calls answer ``already_adopted``) but must not vanish
    silently: one ``extension_generation_adoption_failed`` event row names the
    generation and the error (daemon audit #17 f3: a transient failure poisoned
    the generation with no durable fact anywhere)."""
    import json

    from ouroboros import extension_reconcile_queue as erq

    root = tmp_path / "data"
    (root / "logs").mkdir(parents=True)
    monkeypatch.setattr(erq, "_adopted_generations", {})
    monkeypatch.setattr(erq, "published_extension_generation", lambda _root: "gen-x")
    monkeypatch.setattr("ouroboros.extension_companion.is_server_process", lambda: False)
    monkeypatch.setattr("ouroboros.extension_registry_state.live_extension_fingerprint", lambda: "gen-old")

    def boom(*_a, **_k):
        raise OSError("registry unreadable mid-reload")

    monkeypatch.setattr("ouroboros.extension_loader.reload_all", boom)
    with pytest.raises(OSError):
        erq.adopt_published_extension_generation(root, lambda: {}, repo_path=tmp_path)
    rows = [json.loads(l) for l in (root / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    failed = [r for r in rows if r.get("type") == "extension_generation_adoption_failed"]
    assert len(failed) == 1 and failed[0]["published_generation"] == "gen-x" and "OSError" in failed[0]["error"]
    second = erq.adopt_published_extension_generation(root, lambda: {}, repo_path=tmp_path)
    assert second["action"] == "already_adopted", second  # the contract: one reload, however it ended


@pytest.mark.parametrize("marker", ["absent", "unparseable", "empty_generation"])
def test_an_unreadable_published_registry_is_fail_closed(tmp_path, marker):
    """No readable marker is no evidence of divergence — never a blind reload."""
    _as_worker()
    _loaded, repo_root, drive_root = _prepare_extension(
        tmp_path, "failclosed", _PROBE_PLUGIN, permissions=["tool"])
    path = extension_generation_path(drive_root)
    if marker == "unparseable":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")
    elif marker == "empty_generation":
        atomic_write_json(path, {"schema_version": 1, "generation": ""})

    result = adopt_published_extension_generation(drive_root, lambda: {}, repo_path=str(repo_root))

    assert result == {"action": "no_published_generation"}, result
    assert extension_loader.snapshot()["extensions"] == []


def test_the_steady_state_probe_reads_one_small_file_and_reloads_nothing(
        tmp_path, monkeypatch):
    """The overhead budget: O(read of a small file), no reload, no IPC, no sleep."""
    _loaded, repo_root, drive_root, _surface, _empty, published = (
        _published_by_a_server_that_loaded_it(tmp_path))
    _as_worker()
    monkeypatch.setattr(extension_loader, "reload_all", lambda *a, **kw: pytest.fail(
        "the in-sync steady state must not reload"))

    for _ in range(5):
        result = adopt_published_extension_generation(
            drive_root, lambda: {}, repo_path=str(repo_root))
        assert result == {"action": "in_sync", "generation": published}, result

    assert extension_generation_path(drive_root).stat().st_size < 512


def test_the_server_publishes_and_never_adopts(tmp_path, monkeypatch):
    """The direction switch is the process role, never the caller's opinion."""
    _loaded, repo_root, drive_root, _surface, _empty, _published = (
        _published_by_a_server_that_loaded_it(tmp_path))
    monkeypatch.setattr(extension_loader, "reload_all", lambda *a, **kw: pytest.fail(
        "the publisher adopted its own publication"))

    assert adopt_published_extension_generation(
        drive_root, lambda: {}, repo_path=str(repo_root)) == {"action": "server_process"}
