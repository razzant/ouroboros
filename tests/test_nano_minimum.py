"""A Nano root's normal sender seals a count-only floor, without dispatching either copy."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import context, llm_local, local_model, nano_minimum
from ouroboros import usage_accounting as ua
from ouroboros.chronicle_store import ChronicleStore
from ouroboros.llm import LLMClient
from ouroboros.local_model_server import input_fingerprint
from ouroboros.send_clock import CLOCK_NOTE_PREFIX, SendClockPolicy, main_send_scope
from tests._memory_view_context import world

PORT = 8767
WINDOW = 80_000
RAW = 80_128
TOOLS = [{"type": "function", "function": {"name": "memory_read", "parameters": {"type": "object"}}}]
OWNER_INPUT = "Оперативные слова владельца и назначение задачи 🌱."
WORK = "A bounded result: Fixture owner's standing rules. Keep this occurrence of the same wording."
COUNTER = {"source": "serving_llama_formatter", "formatter": "FixtureFormatter", "library_version": "fixture",
           "chat_format": "fixture-v1", "launched_n_ctx": WINDOW}


@pytest.fixture
def root_plan(tmp_path, monkeypatch):
    from ouroboros import config, deployment_policy

    env, memory, rooms = world(tmp_path)
    ChronicleStore(memory.drive_root).ensure_activated()
    task = {"type": "task", "id": "root-nano", "chat_id": rooms["alpha"], "text": OWNER_INPUT,
            "context_requires_development": True}
    normal = config.runtime_setting
    monkeypatch.setattr(config, "runtime_setting", lambda key, *a, **kw:
                        "Fixture owner's standing rules." if key == "OUROBOROS_SAFETY_OWNER_RULES" else normal(key, *a, **kw))
    monkeypatch.setattr(deployment_policy, "organization_rules", lambda: "Fixture administrator's standing rules.")
    core = context._capture_context_core(env, memory, task, None, None)
    return context._build_context_fit_plan(env, core, task, preferred_mode="nano", tool_schemas=TOOLS,
        route_resolver=lambda *_a, **_kw: ({"model": "local-model", "provider": "local"}, SimpleNamespace(
            route_fp="fixture", status="asserted", stale=False, window_tokens=WINDOW)))


@pytest.mark.parametrize("failure", [None, "instance", "counter"])
def test_normal_local_send_has_one_generation_and_its_own_floor_manifest(tmp_path, monkeypatch, root_plan, failure):
    rows = [*root_plan.messages_for("nano"), {"role": "assistant", "content": WORK}]
    counted, generated = [], []
    lane = LLMClient(api_key="unused")

    def count(input_payload, *, expected_port):
        assert expected_port == PORT
        counted.append(input_payload)
        # Synthetic deterministic formatter/tokenizer, not a prefilled count; it binds every visible byte.
        data = json.dumps({k: input_payload.get(k) for k in ("messages", "tools", "tool_choice")},
                          ensure_ascii=False, sort_keys=True).encode()
        phase = len(counted)
        return {"supported": not (failure == "counter" and phase == 3), "input_is_exact": True,
                "reason": "fixture_unavailable" if failure == "counter" and phase == 3 else None,
                "input_tokens": len(data), "context_window": WINDOW, "raw_context_window": RAW,
                "native_input_sha256": input_fingerprint(input_payload),
                "process_id": 12 if failure == "instance" and phase == 3 else 11,
                "output_limit_enforced": True, "reasoning_included_in_limit": True,
                "tokenizer_template_provenance": {**COUNTER, "rendered_prompt_sha256": hashlib.sha256(data).hexdigest()}}

    def serving_context_evidence(**_):
        return {"process_id": 11, "context_window": WINDOW, "confirmed": True, "port": PORT}

    monkeypatch.setattr(local_model, "get_manager", lambda: SimpleNamespace(
        serving_context_evidence=serving_context_evidence, measure_prepared_input=count))
    sealed = {}

    def fake_create(**wire):
        generated.append(wire)
        return SimpleNamespace(model_dump=lambda: {"choices": [{"message": {"role": "assistant", "content": "fixture"}}]})

    client = SimpleNamespace(base_url=f"http://127.0.0.1:{PORT}/v1",
                             chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    monkeypatch.setattr(lane, "_get_local_client", lambda: client)
    monkeypatch.setattr(llm_local, "limit_physical_send", lambda _, send: send)

    def fake_execute(request, send, before):
        sealed["manifest"] = before(SimpleNamespace(attempt_id="nano-one", drive_root=tmp_path))
        sealed["request"] = request
        assert ua.current_physical_attempt_predicate() is predicate
        return send()

    monkeypatch.setattr(llm_local, "_execute_candidate", fake_execute)
    physical = ua.PhysicalAttemptContext("owner_nano", "nano", "cold_estimate", "fixture", "one", WINDOW,
                                         WINDOW, False, False, preferred_mode="nano",
                                         nano_floor_plan=(root_plan, "cold_start"))
    predicate = lambda _request: True  # noqa: E731
    with ua.bind_physical_attempt_context(physical, candidate_predicate=predicate), main_send_scope(SendClockPolicy()):
        answer, _ = lane._chat_local(rows, TOOLS, 16_000, "auto")
    assert answer["content"] == "fixture"
    assert len(generated) == 1
    with open(sealed["manifest"]["path"], encoding="utf-8") as source:
        manifest = json.load(source)
    result = manifest["nano_minimum"]
    assert result["growth_guard"] == "not_enforced"
    assert result["scope"] == "cold_start"
    assert result["work_headroom_basis"] == "additional_after_this_prepared_work_tail"
    assert result["actual"]["native_input_sha256"] == input_fingerprint(counted[0])
    assert input_fingerprint(generated[0]) == result["actual"]["native_input_sha256"]
    assert sealed["request"].candidate_raw_sha256 == manifest["model_send_seal"]["pre_redaction_sha256"]
    assert len(counted) == 3, result.get("reason")
    n, b = counted[-2:]
    for key in TOOLS[0]["function"]:
        assert n["tools"][0]["function"][key] == b["tools"][0]["function"][key]
    assert n["messages"][1:] == b["messages"][1:]
    assert n["messages"][-2] == {"role": "assistant", "content": WORK}
    assert n["messages"][-1] == generated[0]["messages"][-1]
    assert n["messages"][-1]["content"].startswith(CLOCK_NOTE_PREFIX)
    assert any(OWNER_INPUT in row["content"] for row in b["messages"])
    before, after = (body["messages"][0]["content"] for body in (n, b))
    assert "Fixture owner's standing rules." in before and "Fixture owner's standing rules." not in after
    assert "Fixture administrator's standing rules." in before and "Fixture administrator's standing rules." in after
    assert "## BIBLE.md" in after
    if failure is None:
        assert result["status"] == "exact"
        assert result["memory_sections"] == list(nano_minimum.MEMORY_SECTIONS)
        assert result["minimum"]["native_input_sha256"] == input_fingerprint(n)
        assert result["without_memory"]["native_input_sha256"] == input_fingerprint(b)
        tokens = result["minimum"]["input_tokens"]
        assert result["memory_tokens"] == tokens - result["without_memory"]["input_tokens"] > 0
        assert result["context_window"] == WINDOW and result["raw_context_window"] == RAW
        assert result["minimum_work_headroom_at_reply_floor_tokens"] == WINDOW - tokens - result["reply_floor_tokens"]
        assert result["minimum_work_headroom_at_wire_allowance_tokens"] == WINDOW - tokens - result["minimum_reply_allowance_tokens"]
    else:
        assert result["status"] == "unknown"
        assert result["reason"] == {"instance": "without_memory:serving_binding_changed",
                                    "counter": "without_memory:fixture_unavailable"}[failure]
        assert "memory_tokens" not in result
        assert result["minimum"] == {"native_input_sha256": input_fingerprint(n), "input_tokens": None}
        assert result["without_memory"] == {"native_input_sha256": input_fingerprint(b), "input_tokens": None}


def test_floor_does_not_use_another_snapshot(root_plan):
    rows = root_plan.messages_for("nano")
    rows[0]["content"] = "different system"
    assert nano_minimum.render_minimum(root_plan, rows, window=WINDOW, tool_names=()) == (None, "unbound_system_projection")
    plan = replace(root_plan, core=replace(root_plan.core, memory_view_json=""))
    assert nano_minimum.render_minimum(plan, rows, window=WINDOW, tool_names=()) == (None, "memory_snapshot_unavailable")


@pytest.mark.parametrize(("cold", "predecessor", "expected"), [
    (False, "", "cold_start"), (False, "t-prior", "continue"),
    (True, "", "cold_resume"), (True, "t-prior", "cold_resume_of_continue"),
])
def test_round_gate_counts_only_entry_round(monkeypatch, root_plan, cold, predecessor, expected):
    from ouroboros import body_candidate, loop_round_limits

    slot = SimpleNamespace(task_metadata={}, **{loop_round_limits.COLD_RETURN_ATTR: cold})
    round_ctx = SimpleNamespace(active_use_local=True, task_type="task", tools=SimpleNamespace(_ctx=slot),
                                accumulated_usage={}, round_idx=7)
    monkeypatch.setattr(body_candidate, "continuation_predecessor", lambda _slot: predecessor)
    monkeypatch.setattr(loop_round_limits, "start_unread", lambda *_: not cold)
    assert nano_minimum.measurement_episode(round_ctx, root_plan) == expected
    assert nano_minimum.measurement_episode(round_ctx, root_plan) == expected  # same round's candidate re-fit
    round_ctx.round_idx += 1
    assert nano_minimum.measurement_episode(round_ctx, root_plan) == ""
    assert nano_minimum.measurement_episode(round_ctx, replace(root_plan, preferred_mode="max")) == ""


def test_floor_refuses_to_price_an_owner_input_pointer_as_the_words(root_plan):
    plan = replace(root_plan, nano_projection=replace(root_plan.nano_projection, user_content_json='"addressed"'))
    assert nano_minimum.render_minimum(plan, plan.messages_for("nano"), window=WINDOW, tool_names=()) == (
        None, "owner_input_already_addressed")


@pytest.mark.parametrize(('full_cost', 'form', 'short_world'), [
    (57_000, 'whole', False), (75_000, 'outline', False), (75_000, 'whole', True),
])
def test_full_index_counts_for_publication_even_when_the_physical_floor_uses_its_outline(root_plan, full_cost, form, short_world):
    from ouroboros.nano_publication import publication_status
    from ouroboros.section_floor import HostSection

    science = 'Index author summary.'
    outline = 'Known active note: facts/one.'
    index = HostSection('knowledge_index', ('# Complete index\n' + science + '\n') * 100, outline)
    original = root_plan.core
    sections = tuple(part for part in original.dynamic_head_sections if part.section_id != 'knowledge_index')
    tools = TOOLS
    if short_world:
        world = HostSection('world', 'A lamp lights the observatory. ' * 100, 'Read the world map.')
        original = replace(original, semi_stable_sections=tuple(
            part for part in original.semi_stable_sections if part.section_id != 'world') + (world,))
        tools = [*TOOLS, {'type': 'function', 'function': {'name': 'read_file', 'parameters': {'type': 'object'}}}]
    plan = replace(root_plan, core=replace(original, dynamic_head_sections=(*sections, index)))
    rows = plan.messages_for('nano')
    prepared = []

    def prepare(rendered):
        return {'model': 'local-model', 'messages': rendered, 'tools': tools, 'max_tokens': 16_000}

    def count(payload):
        prepared.append(payload)
        wire = json.dumps(payload, ensure_ascii=False).encode()
        system = json.dumps(payload['messages'][0], ensure_ascii=False)
        value = full_cost if science in system else 54_000 if outline in system else 36_000
        if short_world and 'A lamp lights the observatory.' not in system:
            value -= 10_000
        return {'supported': True, 'input_is_exact': True, 'input_tokens': value,
                'context_window': WINDOW, 'raw_context_window': RAW, 'native_input_sha256': input_fingerprint(payload),
                'process_id': 11, 'output_limit_enforced': True, 'reasoning_included_in_limit': True,
                'tokenizer_template_provenance': {**COUNTER, 'rendered_prompt_sha256': hashlib.sha256(wire).hexdigest()}}

    actual = prepare(rows)
    sample = nano_minimum.measure_minimum(SimpleNamespace(profile='owner_nano', nano_floor_plan=(plan, 'cold_start')),
        rows, actual, count(actual), 16_000, prepare, count)
    assert sample['status'] == 'exact' and sample['knowledge_index_form'] == form, sample
    shift = 10_000 if short_world else 0
    n = full_cost - shift if form == 'whole' else 54_000
    assert sample['world_form'] == ('reader' if short_world else 'whole')
    assert sample['minimum']['input_tokens'] == n and sample['without_memory']['input_tokens'] == 36_000 - shift
    assert sample['memory_tokens'] == n - 36_000 + shift and sample['whole_memory_tokens'] == full_cost - 36_000
    assert sample['minimum_work_headroom_at_reply_floor_tokens'] == WINDOW - n - sample['reply_floor_tokens']
    assert publication_status(provider='local', measurement=sample)['pressure']['memory_excess_tokens'] == full_cost - 56_000
    assert all(payload['messages'][1:] == rows[1:] for payload in prepared)


@pytest.fixture
def publication_world(tmp_path, monkeypatch):
    """Full context builder and real writers; only the owned count service is a double."""
    from ouroboros import config, model_concurrency
    from ouroboros.nano_publication import local_root_publication, source_bytes
    from ouroboros.utils import atomic_write_json, machine_json
    from contextlib import nullcontext

    env, memory, rooms = world(tmp_path)
    memory.ensure_files()
    ChronicleStore(memory.drive_root).ensure_activated()
    memory.identity_path().write_text("An exact, ordinary identity.", encoding="utf-8")
    atomic_write_json(memory.scratchpad_blocks_path(), [])  # the shared fixture starts with a pre-block scratchpad
    memory.append_scratchpad_block("An existing block.")
    task = {"type": "task", "id": "publication-root", "chat_id": rooms["alpha"],
            "delegation_role": "root", "text": "Do not price this writer's current work."}
    original = config.runtime_setting
    monkeypatch.setattr(config, "runtime_setting", lambda key, *args, **kwargs:
                        "Operative fixture rules." if key == "OUROBOROS_SAFETY_OWNER_RULES" else original(key, *args, **kwargs))
    monkeypatch.setattr(context, "_context_fit_route", lambda *_a, **_kw: (
        {"model": "local-model", "provider": "local"}, SimpleNamespace(route_fp="fixture-root",
                status="asserted", stale=False, window_tokens=WINDOW)))
    live = {"scope": "r1", "pid": 11, "offset": 0, "counted": [], "on_count": None}

    def count(payload, *, expected_port):
        assert expected_port == PORT
        text = machine_json({k: payload[k] for k in ("messages", "tools", "tool_choice") if k in payload})
        if live["on_count"]:
            live["on_count"]()
        live["counted"].append(text)
        # All bytes contribute through framing, plus a dense synthetic sequence.
        # A conditional marker here is a deterministic tokenizer fixture, not a provider heuristic.
        dense = text.count("COUNTED_GROWTH_700") * 700
        return {"supported": True, "input_is_exact": True, "input_tokens": len(text.encode()) // 4096 + dense + live["offset"],
                "context_window": WINDOW, "raw_context_window": RAW, "native_input_sha256": input_fingerprint(payload),
                "process_id": live["pid"], "output_limit_enforced": True, "reasoning_included_in_limit": True,
                "tokenizer_template_provenance": {**COUNTER, "rendered_prompt_sha256": hashlib.sha256(text.encode()).hexdigest()}}

    def serving_context_evidence(**_):
        return {"process_id": live["pid"], "context_window": WINDOW, "confirmed": True, "port": PORT}

    monkeypatch.setattr(local_model, "get_manager", lambda: SimpleNamespace(
        serving_context_evidence=serving_context_evidence, measure_prepared_input=count))
    monkeypatch.setattr(local_model, "local_dispatch_port", lambda: PORT)
    monkeypatch.setattr(model_concurrency, "measurement_slot", lambda *a, **kw: nullcontext(True))
    monkeypatch.setattr(llm_local, "measurement_slot", lambda *a, **kw: nullcontext(True))
    client = LLMClient(api_key="unused")
    monkeypatch.setattr(client, "_get_local_client", lambda: pytest.fail("A publication cannot open the inference client"))
    author = SimpleNamespace(drive_root=memory.drive_root, task_id="memory-author")
    admission = local_root_publication(env, memory, task, author_ctx=author, client=client,
                                  tool_schemas=TOOLS, caller_max=16_000, scope_revision=lambda: live["scope"])
    from ouroboros.nano_publication import MemoryChange

    raw = source_bytes(memory.identity_path())
    baseline, _ = admission.capture((MemoryChange(memory.identity_path(), raw, raw),))
    sample = baseline.count()
    assert sample["status"] == "exact", sample
    live["offset"] = WINDOW - sample["reply_floor_tokens"] - 15_000 - 900 - sample["minimum"]["input_tokens"]
    live["counted"].clear()
    return memory, admission, author, live


def test_publication_bracket_uses_one_mcp_map_despite_catalogue_drift(publication_world, monkeypatch):
    from ouroboros import mcp_client
    from ouroboros.nano_publication import MemoryChange
    from ouroboros import tool_connections

    memory, admission, _, _ = publication_world
    raw = memory.identity_path().read_bytes()
    original = context._connections
    observations = []
    namespace = "mcp_lantern"
    tool = {"type": "function", "function": {"name": namespace + "__report", "parameters": {"type": "object"}}}
    monkeypatch.setattr(mcp_client, "get_manager", lambda: SimpleNamespace(
        list_tools_for_registry=lambda: [{"name": namespace + "__report", "server_id": "lantern",
                                         "raw_description": "Look up a lamp."}],
        server_facts=lambda: {"lantern": {"instructions": "Open the red box." if len(observations) == 1
                                         else "Open the green box."}},
    ))

    def connections(registry, *args, **kwargs):
        observations.append(registry)
        assert registry is None
        double = SimpleNamespace(schemas=lambda: [tool])
        return original(double, *args, **kwargs)

    monkeypatch.setattr(context, "_connections", connections)
    monkeypatch.setattr(tool_connections, "usable_selectors", lambda _: ())
    before, proposed = admission.capture((MemoryChange(memory.identity_path(), raw, b"Proposed pencil identity."),))
    assert len(observations) == 2
    for frame in (before, proposed):
        blob = json.dumps(frame.actual, ensure_ascii=False)
        assert "Open the red box." in blob and "Open the green box." not in blob
    assert "Proposed pencil identity." in json.dumps(proposed.actual, ensure_ascii=False)


def test_count_only_publication_reaches_formatter_despite_large_estimated_core(publication_world, monkeypatch):
    from ouroboros import context_start_bound
    from ouroboros.nano_publication import MemoryChange

    memory, admission, _, _ = publication_world
    raw = memory.identity_path().read_bytes()
    # The second build would fail the 80k estimator's bound, although its
    # candidate is merely being priced, not admitted for dispatch.
    monkeypatch.setattr(context_start_bound, "_lowest_density", lambda *_: 1.0)
    candidate = b"Dense estimate but compact formatter. " + b"hydrogen " * 40_000
    _, proposed = admission.capture((MemoryChange(memory.identity_path(), raw, candidate),))
    assert "Dense estimate but compact formatter." in json.dumps(proposed.actual, ensure_ascii=False)
    assert proposed.count()["status"] == "exact"


def _identity_edit(memory, text, admission):
    from ouroboros.memory_publication import publish_text
    from ouroboros.memory_texts import identity_source

    source = identity_source(memory.drive_root)
    revision = hashlib.sha256(source.path.read_bytes()).hexdigest()
    return publish_text(source, revision, lambda _: text,
                        author={"task_id": "memory-author", "focus": {}, "route": "fixture"},
                        source_type="task", mode="overwrite", nano_admission=admission)


def test_publication_counts_the_complete_candidate_and_preserves_refused_growth(publication_world):
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.nano_publication import NanoPublicationRefused

    memory, admission, author, live = publication_world
    growth = "Add one new memory: COUNTED_GROWTH_700"
    result = _identity_edit(memory, growth, admission)
    assert result.ok
    original = memory.scratchpad_blocks_path().read_bytes()
    with pytest.raises(NanoPublicationRefused, match="over_capacity") as refusal:
        memory.append_scratchpad_block(growth, nano_admission=admission)
    assert memory.scratchpad_blocks_path().read_bytes() == original
    assert refusal.value.reader
    retained = list(task_artifact_dir_path(author.drive_root, author.task_id).rglob("*nano-proposed-memory*.json"))
    assert len(retained) == 1
    attempted = json.loads(retained[0].read_text())
    assert len(attempted) == 1 and growth in attempted[0]["text"]
    assert any("Work starts here." in wire and "Operative fixture rules." in wire for wire in live["counted"])
    assert not any("Do not price this writer's current work." in wire for wire in live["counted"])
    assert any(wire.count("COUNTED_GROWTH_700") == 2 for wire in live["counted"])
    assert _identity_edit(memory, "A genuine capacity recovery.", admission).ok
    memory.append_scratchpad_block(growth, nano_admission=admission)
    assert len(memory.load_scratchpad_blocks()) == 2


def test_two_memory_writers_cannot_spend_the_same_remaining_room(publication_world):
    from threading import Barrier, Thread, current_thread, local
    from ouroboros.nano_publication import NanoPublicationRefused

    memory, admission, _, live = publication_world
    gate, once, outcomes = Barrier(2, timeout=20), local(), {}

    def arrive():
        if not getattr(once, "done", False):
            once.done = True
            gate.wait()

    def writer():
        name = current_thread().name
        try:
            if name == "identity":
                result = _identity_edit(memory, "An identity COUNTED_GROWTH_700.", admission)
                outcomes[name] = "saved" if result.ok else result.reason.removeprefix("nano_")
            else:
                memory.append_scratchpad_block("A block COUNTED_GROWTH_700.", nano_admission=admission)
                outcomes[name] = "saved"
        except NanoPublicationRefused as refusal:
            outcomes[name] = refusal.reason

    live["on_count"] = arrive
    writers = [Thread(target=writer, name=name) for name in ("identity", "scratchpad")]
    for thread in writers:
        thread.start()
    for thread in writers:
        thread.join(25)
    assert not any(thread.is_alive() for thread in writers)
    assert sorted(outcomes.values()) == ["saved", "stale"]
    assert sum("COUNTED_GROWTH_700" in content for content in (memory.load_identity(),
                                   str(memory.load_scratchpad_blocks()))) == 1


def test_interruption_before_activation_leaves_original_and_no_count_credit(publication_world, monkeypatch):
    from ouroboros import memory_publication
    from ouroboros.memory import Memory

    memory, admission, _, _ = publication_world
    identity = memory.identity_path().read_bytes()
    replacement = memory_publication.write_bytes_atomic

    class Interrupted(BaseException):
        pass

    def interrupt(path, raw):
        assert path == memory.identity_path() and b"COUNTED_GROWTH_700" in raw
        raise Interrupted("stopped immediately before atomic replacement")

    with monkeypatch.context() as patch:
        patch.setattr(memory_publication, "write_bytes_atomic", interrupt)
        with pytest.raises(Interrupted):
            _identity_edit(memory, "A captured edit COUNTED_GROWTH_700.", admission)
    assert memory_publication.write_bytes_atomic is replacement
    cold = Memory(drive_root=memory.drive_root)
    assert cold.load_identity().encode() == identity
    assert b"COUNTED_GROWTH_700" in memory.identity_journal_path().read_bytes()
    assert _identity_edit(cold, "A different edit COUNTED_GROWTH_700.", admission).ok
    assert "different edit" in memory.load_identity()


def test_aba_of_active_memory_invalidates_an_inflight_capture(publication_world):
    import subprocess
    import sys

    memory, admission, _, live = publication_world
    path = memory.scratchpad_blocks_path()
    prior = path.read_bytes()
    child = """
from pathlib import Path
import json
import sys
sys.path.insert(0, sys.argv[1])
from ouroboros.memory import Memory
from ouroboros.nano_publication import publication_fence
memory = Memory(drive_root=Path(sys.argv[2]))
path = memory.scratchpad_blocks_path()
prior = path.read_bytes()
memory.append_scratchpad_block('A concurrently visible, then superseded block.')
with publication_fence(memory.drive_root):
    path.write_bytes(prior)
    memory._write_scratchpad_markdown(json.loads(prior))
"""

    def change_and_restore():
        live["on_count"] = None
        subprocess.run([sys.executable, "-c", child, str(Path(__file__).resolve().parent.parent),
                        str(memory.drive_root)], check=True, timeout=10)

    identity = memory.load_identity()
    live["on_count"] = change_and_restore
    refused = _identity_edit(memory, "A distinct proposal COUNTED_GROWTH_700.", admission)
    assert not refused.ok and refused.reason == "nano_stale"
    assert path.read_bytes() == prior and memory.load_identity() == identity


def test_publication_binds_one_bootstrapped_installation(publication_world, tmp_path, monkeypatch):
    from ouroboros.nano_publication import MemoryChange, NanoPublicationRefused, admission_for

    memory, admission, _, live = publication_world
    other = tmp_path / "another-installation"
    other.mkdir()
    monkeypatch.chdir(other)
    assert admission_for(None, other) is None  # never lend an installation counter to another profile
    for invalid in (Path("identity.md"), other / "identity.md"):
        with pytest.raises(NanoPublicationRefused, match="stale"):
            with admission.publication(MemoryChange(invalid, None, b"A proposal.")):
                pytest.fail("cannot publish to an unbound path")
    assert not live["counted"]
    memory.world_path().unlink()  # synthetic profile lost an ordinary input; a writer must not seed it
    prior = memory.identity_path().read_bytes()
    refused = _identity_edit(memory, "A proposal before boot.", admission)
    assert refused.reason == "nano_bootstrap_pending"
    assert not memory.world_path().exists() and memory.identity_path().read_bytes() == prior
    assert not live["counted"]


@pytest.mark.parametrize("provider, sample, reason", [
    ("ollama", None, "unsupported_route"),
    ("local", None, "exact_counter_unproven"),
    ("local", {"status": "exact", "context_window": 80_000}, "publication_scope_unqualified"),
    ("local", {"status": "exact", "context_window": 120_000}, "outside_qualified_window"),
])
def test_publication_status_does_not_treat_observation_as_qualification(provider, sample, reason):
    from ouroboros.nano_publication import publication_status

    assert publication_status(provider=provider, measurement=sample) == {
        "status": "not_claimed", "reason": reason, "measurement": "exact" if sample else "unavailable",
        "current_sample": "unavailable", "pressure": None,
    }


def test_exact_samples_can_be_inside_or_outside_without_a_future_guarantee():
    from ouroboros.nano_publication import publication_status

    base = {"status": "exact", "context_window": 80_000, "minimum_work_headroom_at_reply_floor_tokens": 15_000}
    inside = publication_status(provider="local", measurement={**base, "memory_tokens": 20_000})
    outside = publication_status(provider="local", measurement={**base, "memory_tokens": 22_404})
    assert inside["current_sample"] == "within_limits" and outside["current_sample"] == "outside_limits"
    assert outside["pressure"] == {"memory_excess_tokens": 2404, "prepared_overflow_tokens": 0}
    assert inside["status"] == outside["status"] == "not_claimed"


def test_work_text_already_in_a_measured_main_input_is_not_a_work_shortfall():
    """Headroom is room beyond the work text the sample already carries (``nano_minimum.measure_minimum``).

    Background 34,000 + memory 19,000 + work text 15,000 = 68,000; at the 8,192 reply floor 3,808 more stay free.
    The 15,000 work floor is met, but the sample cannot separate work text from background, so it is unconfirmed,
    neither a 11,192-token shortfall nor confirmed compliance. Memory excess and a physical overflow stay outside."""
    from ouroboros.nano_publication import _pressure, publication_status

    window, reply = 80_000, 8_192
    sample = {"status": "exact", "context_window": window, "memory_tokens": 19_000, "reply_floor_tokens": reply,
              "minimum_work_headroom_at_reply_floor_tokens": window - (34_000 + 19_000 + 15_000) - reply}
    assert sample["minimum_work_headroom_at_reply_floor_tokens"] == 3_808
    assert _pressure(sample) == (0, 11_192)  # a writer's neutral opening keeps this meaning for admission
    status = publication_status(provider="local", measurement=sample)
    assert status["current_sample"] == "work_floor_unconfirmed"
    assert status["pressure"] == {"memory_excess_tokens": 0, "prepared_overflow_tokens": 0}
    over = publication_status(provider="local", measurement={**sample, "minimum_work_headroom_at_reply_floor_tokens": -1_200})
    assert over["current_sample"] == "outside_limits"
    assert over["pressure"] == {"memory_excess_tokens": 0, "prepared_overflow_tokens": 1_200}
    both = publication_status(provider="local", measurement={**sample, "whole_memory_tokens": 21_000})
    assert both["current_sample"] == "outside_limits" and both["pressure"]["memory_excess_tokens"] == 1_000


def test_root_measurement_summary_is_bounded_and_does_not_outlive_its_serving_binding(tmp_path, monkeypatch):
    from ouroboros import config, nano_host_publication
    from ouroboros.nano_publication import read_root_sample, record_root_sample

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(nano_host_publication, "current_rules_digest", lambda: "rules-fixture")
    live = {"confirmed": True, "process_id": 11, "context_window": 80_000, "port": PORT}
    monkeypatch.setattr(local_model, "get_manager", lambda: SimpleNamespace(serving_context_evidence=lambda **_: live))
    floor = {"status": "exact", "scope": "cold_start", "room": "main", "process_id": 11, "context_window": 80_000,
             "port": PORT, "rules_sha256": "rules-fixture",
             "memory_tokens": 22_404, "minimum": {"input_tokens": 58_056},
             "reply_floor_tokens": 8192, "minimum_reply_allowance_tokens": 8192,
             "minimum_work_headroom_at_reply_floor_tokens": 13_752,
             "minimum_work_headroom_at_wire_allowance_tokens": 13_752,
             "actual": {"sent_max_tokens": 8192, "input_tokens": 60_000, "body": "must not be copied"},
             "body": "must not be copied"}
    record_root_sample(tmp_path, {**floor, "scope": "publication_candidate"})
    record_root_sample(tmp_path, {**floor, "scope": "cold_resume", "held_sections": ["shared_understanding"]})
    record_root_sample(tmp_path, {**floor, "room": "project"})
    assert read_root_sample(tmp_path, provider="local")["measurement"] == "unavailable"
    record_root_sample(tmp_path, floor)
    status = read_root_sample(tmp_path, provider="local")
    assert status["scope"] == "cold_start" and status["observed_at"] and status["observation"] == "prepared"
    assert status["minimum_input_tokens"] == 58_056 and status["sent_max_tokens"] == 8192
    assert status["status"] == "not_claimed" and status["current_sample"] == "outside_limits"
    assert "must not be copied" not in (tmp_path / "state/nano_minimum_latest.json").read_text()
    assert "_process_id" not in status and "minimum_input_tokens" not in read_root_sample(tmp_path, provider="ollama")
    live["process_id"] = 12
    changed = read_root_sample(tmp_path, provider="local")
    assert changed["reason"] == "serving_binding_changed" and "minimum_input_tokens" not in changed


def test_world_reader_keeps_its_form_when_the_index_also_outlines_and_records_omitted_tokens(root_plan):
    from ouroboros.section_floor import HostSection

    science = "Index author summary."
    outline = "Known active note: facts/one."
    index = HostSection("knowledge_index", ("# Complete index\n" + science + "\n") * 100, outline)
    world_section = HostSection("world", "A lamp lights the observatory. " * 100, "Read the world map.")
    original = root_plan.core
    head = tuple(part for part in original.dynamic_head_sections if part.section_id != "knowledge_index")
    semi = tuple(part for part in original.semi_stable_sections if part.section_id != "world")
    plan = replace(root_plan, core=replace(original, semi_stable_sections=(*semi, world_section),
                                           dynamic_head_sections=(*head, index)))
    tools = [*TOOLS, {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}]
    rows = plan.messages_for("nano")
    prepared = []

    def prepare(rendered):
        return {"model": "local-model", "messages": rendered, "tools": tools, "max_tokens": 16_000}

    def count(payload):
        prepared.append(payload)
        wire = json.dumps(payload, ensure_ascii=False).encode()
        system = json.dumps(payload["messages"][0], ensure_ascii=False)
        value = 84_000 if science in system else 66_000 if outline in system else 36_000
        if "A lamp lights the observatory." not in system:
            value -= 10_000
        return {"supported": True, "input_is_exact": True, "input_tokens": value,
                "context_window": WINDOW, "raw_context_window": RAW, "native_input_sha256": input_fingerprint(payload),
                "process_id": 11, "output_limit_enforced": True, "reasoning_included_in_limit": True,
                "tokenizer_template_provenance": {**COUNTER, "rendered_prompt_sha256": hashlib.sha256(wire).hexdigest()}}

    actual = prepare(rows)
    sample = nano_minimum.measure_minimum(SimpleNamespace(profile="owner_nano", nano_floor_plan=(plan, "cold_start")),
                                          rows, actual, count(actual), 16_000, prepare, count)
    assert sample["status"] == "exact", sample
    assert (sample["world_form"], sample["knowledge_index_form"]) == ("reader", "outline")
    assert sample["world_omitted_tokens"] == 10_000
    assert (sample["minimum"]["input_tokens"], sample["without_memory"]["input_tokens"]) == (56_000, 26_000)
    assert (sample["memory_tokens"], sample["whole_memory_tokens"]) == (30_000, 48_000)
    assert len(prepared) == 5

    prepared.clear()
    publication = SimpleNamespace(profile="owner_nano", nano_floor_plan=(plan, "publication_candidate"))
    fixed = nano_minimum.measure_minimum(publication, rows, actual, count(actual), 16_000, prepare, count)
    assert fixed["status"] == "exact", fixed
    assert (fixed["world_form"], fixed["knowledge_index_form"]) == ("whole", "outline")
    assert (fixed["minimum"]["input_tokens"], fixed["without_memory"]["input_tokens"]) == (66_000, 36_000)
    assert (fixed["memory_tokens"], fixed["whole_memory_tokens"]) == (30_000, 48_000)
    assert "world_omitted_tokens" not in fixed
