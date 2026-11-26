"""A refused successor never installs its tool list on the still-serving call."""
import asyncio
import copy
import json
from dataclasses import replace

from ouroboros.contracts.live_conversation import LiveOptions, ToolOutputs
from tests._extension_loader_shared import _clear_loader_state  # noqa: F401
from tests._live_world import host_world as host_world


SMALL = ("memory_read knowledge_read knowledge_list journal_read workpad_read read_file list_files "
         "query_data chat_history recent_tasks live_roots memory_map list_projects").split()


def test_rejected_large_catalog_does_not_follow_a_smaller_request(host_world):
    from ouroboros.realtime_setup import measure_session_wire
    from skills.live_conversation.driver import _Call, session_options

    host = host_world.host
    def names(schemas):
        return [row["function"]["name"] for row in schemas]
    catalog = {s["function"]["name"]: s for s in host.tools.schemas()}
    # A tool acquired before this transaction is a real part of the first serving setup.
    if "search_code" not in names(host.schemas):
        host.schemas.append(catalog["search_code"])
    host.ctx._actor_loaded_tool_names.add("search_code")
    note = "The working lighthouse file was read in the adopted view."
    (host_world.env.repo_dir / "lighthouse.txt").write_text(note, encoding="utf-8")
    async def scenario(session):
        call = _Call(host)
        call.session, call.conversation = session, 1
        call.connected.set()
        serving = list(call.serving_catalog())
        previous = names(serving)
        assert "search_code" in previous
        small = SMALL
        assert set(small) <= catalog.keys() and set(small).isdisjoint(previous)
        large = (small + [n for n in sorted(catalog) if n not in previous and n not in small])[:116]
        assert len(large) > 40
        def shape(n):
            return [*serving, *(catalog[k] for k in n)]
        opt = session_options(host)
        def count(n):
            return measure_session_wire(host, host.plan, host.messages, shape(n),
                host_world.provider["session"], opt, _room_fact=False)[2]["fit"]["measurement"]["estimated_input_tokens"]
        # Exercise both sides of the real full-envelope admission, without a paid provider or a dummy size refusal.
        less, more = count(small), count(large)
        assert more > less + 5000
        window = (less + more) // 2 + 10_000
        host.model = replace(host.model, facts=replace(host.model.facts, conversation_window_tokens=window))
        sent = []
        async def send(output): sent.append(output)
        session.send = send
        # As on the serving Grok socket, an ordinary growth receipt triggers the fit even for an engine
        # that can update tools live. The smaller set still requires a fresh history, which it can carry.
        host._held_input = window - 10_000 - (less - count([])) // 2
        assert not host.renewal_due()
        async def ask(cid, requested):
            await call.tool_queue.put({"id": cid, "type": "function", "function": {"name": "enable_tools",
                                         "arguments": json.dumps({"tools": ",".join(requested)})}})
            await asyncio.wait_for(call.tool_queue.join(), 45)
        worker = asyncio.create_task(call.tools())
        try:
            await ask("large", large)
            assert call.preparing is not None
            await asyncio.gather(call.preparing, return_exceptions=True)
            assert getattr(call.preparing.exception(), "code", None) == "context_capacity"
            await call.check_preparation()
            assert names(host.schemas) == previous
            assert "search_code" in host.ctx._actor_loaded_tool_names
            assert set(large).isdisjoint(host.ctx._actor_loaded_tool_names)
            await call.release_serving()
            output = next(o for o in sent if isinstance(o, ToolOutputs))
            assert names(output.tools) == previous and any(r.get("tool_call_id") == "large" for r in output.messages)
            assert any("did not fit" in str(r.get("content")) for r in output.messages)
            assert any(r.get("_host_context_kind") == "call_tool_residency" for r in host.messages)
            await ask("small", small)
            assert set(names(host.schemas)) == {*previous, *small}
            assert call.preparing is not None
            lease = await asyncio.wait_for(call.preparing, 45)
            assert set(names(lease.view.schemas)) == {*previous, *small}
            async def use(new):
                assert set(names(json.loads(new.selected.tools_json))) == {*previous, *small}
                assert set(names(host.schemas)) == {*previous, *small}
                assert len([m for m in host.messages if m.get("tool_call_id") in ("large", "small")]) == 2
                assert {*small, "search_code"} <= host.ctx._actor_loaded_tool_names
                rows = await host.tool_batch([{"id": "reading", "type": "function", "function": {
                    "name": "read_file", "arguments": json.dumps({"root": "system_repo", "path": "lighthouse.txt"})}}])
                assert note in next(r["content"] for r in rows if r.get("tool_call_id") == "reading")
            await lease.run(use)
            call.preparing = None
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            if call.preparing is not None:
                value = await asyncio.gather(call.preparing, return_exceptions=True)
                if value and getattr(value[0], "close", None): await value[0].close()
    asyncio.run(host.session(scenario, LiveOptions(instructions="exercise real catalog admission")))


def test_a_stale_refusal_cannot_remove_a_newer_tool_choice(host_world):
    host = host_world.host
    original, rev = copy.deepcopy(host.schemas), host._history_version
    addition = next(s for s in host.tools.schemas() if s not in original)
    host.schemas.append(addition)
    request = copy.deepcopy(host.schemas)
    host._context_changed()
    assert asyncio.run(host.restore_serving_catalog(request, original, rev)) is None
    assert host.schemas == request
    # Even with a matching revision the physical catalog comparison is mandatory.
    assert asyncio.run(host.restore_serving_catalog(original, original, host._history_version)) is None
    assert host.schemas == request


def test_an_in_flight_effect_merges_only_its_own_new_tools(host_world, monkeypatch):
    from tests.test_live_tool_continuity import Held, invocation

    host = host_world.host
    held = Held(monkeypatch)
    previous = list(host.schemas)
    pending = next(s for s in host.tools.schemas() if s["function"]["name"] == "query_data")
    assert pending not in previous
    host.schemas.append(pending)
    host.ctx._actor_loaded_tool_names.add("query_data")

    async def scenario():
        work = asyncio.create_task(host.tool_batch([
            invocation("load", "enable_tools", tools="read_file"), invocation("held", "list_available_tools")]))
        try:
            await held.started()
            # Refusal of a snapshot taken with the effect pending: the tool receipt isn't history yet.
            note = await host.restore_serving_catalog(list(host.schemas), previous, host._history_version)
            assert note is not None
        finally:
            held.release.set()
            await work
        assert {s["function"]["name"] for s in host.schemas} == {
            *(s["function"]["name"] for s in previous), "read_file"}
        assert "read_file" in host.ctx._actor_loaded_tool_names and "query_data" not in host.ctx._actor_loaded_tool_names

    asyncio.run(scenario())

def test_native_grok_keeps_serving_until_a_refused_catalog_is_reconciled(host_world, monkeypatch):
    # Everything from the owner's button through the native adapter is real; only the page and remote peer are local.
    from tests._live_stack import stack
    from ouroboros.realtime_setup import measure_session_wire

    async def scenario():
        note = "Blue beacon, loaded from the file after narrowing."
        (host_world.env.repo_dir / "passage.txt").write_text(note, encoding="utf-8")
        async with stack(host_world, monkeypatch, "grok") as live:
            host, call, wire = live.host, live.call, live.wire
            serving = tuple(call.serving_catalog())
            resident = {s["function"]["name"] for s in serving}
            catalog = {s["function"]["name"]: s for s in host.tools.schemas()}
            assert resident.isdisjoint(SMALL) and set(SMALL) <= catalog.keys()
            large = (SMALL + sorted(catalog.keys() - resident - set(SMALL)))[:115]

            def price(names):
                view = host._connection_view
                return measure_session_wire(host, host.plan, host.messages, [*serving, *(catalog[n] for n in names)],
                    view._view_session_factory, view._view_session_options, _room_fact=False)[2]["fit"]["measurement"]["estimated_input_tokens"]

            low, high, initial = await asyncio.to_thread(lambda: (price(SMALL), price(large), price([])))
            reserve = host.model.facts.working_reserve_tokens
            window = (low + high) // 2 + reserve
            host.model = replace(host.model, facts=replace(host.model.facts, conversation_window_tokens=window))
            host._held_input = window - reserve - (low - initial) // 2  # observed serving usage
            assert not host.renewal_due()
            restore = host.restore_serving_catalog

            async def reconcile(*args):
                assert not call.preparation_ready(), "the socket may retire while the host records the refusal"
                return await restore(*args)

            monkeypatch.setattr(host, "restore_serving_catalog", reconcile)
            wire.tool("large", "enable_tools", {"tools": ",".join(large)})
            await live.wait(lambda: any("did not fit" in json.dumps(v) for v in wire.sent))
            assert live.provider.connections == [wire] and tuple(host.schemas) == serving
            await live.quiet()
            wire.tool("smaller", "enable_tools", {"tools": ",".join(SMALL)})
            expected = resident | set(SMALL)
            await live.wait(lambda: any({t["name"] for t in event.get("session", {}).get("tools", [])} == expected
                for connection in live.provider.connections for event in connection.sent if event.get("type") == "session.update"))
            await live.quiet()
            await live.tool("reading", "read_file", {"root": "system_repo", "path": "passage.txt"})
            await live.wait(lambda: any(note in json.dumps(v) for v in live.wire.sent))
            assert {s["function"]["name"] for s in host.schemas} == expected
        assert live.result[1]["execution_status"] == "ok"

    asyncio.run(scenario())


def test_a_permitted_tool_runs_once_after_the_rollback(host_world, monkeypatch):
    """After a refused larger catalogue is rolled back, a tool of the serving set is called, runs, and its effect
    happens exactly once."""
    from tests.test_live_tool_continuity import Held, invocation

    host = host_world.host
    held = Held(monkeypatch, block=False)
    serving = copy.deepcopy(host.schemas)
    resident = {s["function"]["name"] for s in serving}
    assert "list_available_tools" in resident and resident.isdisjoint(SMALL)
    host.schemas.extend(s for s in host.tools.schemas() if s["function"]["name"] in SMALL)
    host.ctx._actor_loaded_tool_names.update(SMALL)
    rejected = copy.deepcopy(host.schemas)

    async def scenario():
        note = await host.restore_serving_catalog(rejected, serving, host._history_version)
        assert note is not None and "did not fit" in note["content"]
        assert host.schemas == serving and set(SMALL).isdisjoint(host.ctx._actor_loaded_tool_names)
        return await host.tool_batch([invocation("after", "list_available_tools")])

    asyncio.run(scenario())
    assert held.effects == ["list_available_tools"]
    assert len([row for row in host.messages if row.get("tool_call_id") == "after"]) == 1
