"""Section timing and counted work of one measured window: recorded only inside its scope, errors unchanged."""
from __future__ import annotations

import gc
import threading

import pytest

from ouroboros.context_timing import collector_rests, counted, measured_sections, section_spans, timed, timed_section


@timed_section("demo")
def _builder(value, *, fail=False):
    if fail:
        raise RuntimeError("the section could not be built")
    return value * 2


def test_a_marked_builder_records_nothing_outside_a_measured_build():
    assert _builder(3) == 6
    assert section_spans() == {}


def test_a_measured_build_sums_each_sections_calls_and_keeps_its_errors():
    with measured_sections():
        assert _builder(1) == 2 and _builder(2) == 4
        assert timed("other", lambda: "built") == "built"
        with pytest.raises(RuntimeError, match="could not be built"):
            _builder(0, fail=True)
        spans = section_spans()
    assert set(spans["sections_ms"]) == set(spans["sections_cpu_ms"]) == {"demo", "other", "total"}
    assert spans["sections_ms"]["total"] >= max(spans["sections_ms"]["demo"], spans["sections_ms"]["other"])
    assert section_spans() == {}


def test_another_threads_build_does_not_see_this_ones_spans():
    seen = {}

    def other():
        _builder(1)
        seen["spans"] = section_spans()

    with measured_sections():
        _builder(1)
        worker = threading.Thread(target=other)
        worker.start()
        worker.join()
        assert set(section_spans()["sections_ms"]) == {"demo", "total"}
    assert seen["spans"] == {}


def test_counted_adds_up_inside_a_window_and_records_nothing_outside():
    counted("reads", 7, key="outside")
    assert section_spans() == {}
    with measured_sections():
        assert section_spans()["work"] == {}
        counted("reads")
        counted("reads", 4)
        assert section_spans()["work"] == {"reads": 5}
    assert section_spans() == {}
    with measured_sections():  # the key counted outside any window left no trace in this one
        counted("reads", key="outside")
        assert section_spans()["work"] == {"reads": 1}


def test_again_counts_only_a_repeated_key_of_the_same_verb():
    with measured_sections():
        counted("renders", key="a")
        counted("renders", key="b")
        counted("parses", key="a")  # the same key under another verb is not a repeat
        assert section_spans()["work"] == {"renders": 2, "parses": 1}
        counted("renders", 3, key="a")  # the repeat adds its whole amount to the verb and to _again
        counted("renders", 2)  # work without a key is never a repeat
        assert section_spans()["work"] == {"renders": 7, "parses": 1, "renders_again": 3}


def test_a_window_opened_inside_another_keeps_its_own_counts():
    with measured_sections():
        counted("reads", key="k")
        with measured_sections():  # a round's preparation opens the same window: nothing carries in or out
            counted("reads", key="k")
            assert section_spans()["work"] == {"reads": 1}
        counted("reads", key="k")
        assert section_spans()["work"] == {"reads": 2, "reads_again": 1}


def test_another_threads_window_does_not_see_this_ones_work():
    seen = {}

    def other():
        counted("reads")
        seen["spans"] = section_spans()

    with measured_sections():
        counted("reads")
        worker = threading.Thread(target=other)
        worker.start()
        worker.join()
        assert section_spans()["work"] == {"reads": 1}
    assert seen["spans"] == {}


def test_the_collector_rests_for_a_block_and_resumes_after_it():
    """The collector is off inside the block and back on after it, an exception included; a collector the host already
    switched off stays off."""
    assert gc.isenabled()
    with collector_rests():
        assert not gc.isenabled()
    assert gc.isenabled()
    with pytest.raises(RuntimeError):
        with collector_rests():
            raise RuntimeError("the build failed")
    assert gc.isenabled()
    gc.disable()
    try:
        with collector_rests():
            assert not gc.isenabled()
        assert not gc.isenabled()
    finally:
        gc.enable()


def test_a_collection_inside_the_resting_block_counts_in_its_window_and_nothing_outside_one():
    """A collection that still happens inside a measured build (an explicit ``gc.collect``) is the build's work; the
    collector's own count of it stays out of a window nobody opened."""
    with collector_rests(), measured_sections():
        gc.collect()
        work = section_spans()["work"]
    assert work["gc_collections"] >= 1 and work["gc_ms"] >= 0
    with collector_rests():
        gc.collect()
    assert section_spans() == {}


@pytest.mark.parametrize("realtime", [False, True])
def test_the_agent_measures_its_context_build(tmp_path, monkeypatch, realtime):
    """The agent opens the measured scope around its context build, so the receipt can carry the spans."""
    import pathlib
    import queue
    from types import SimpleNamespace

    from ouroboros import agent as agent_module, context_timing, provider_models
    from ouroboros.agent import Env, OuroborosAgent

    seen = []
    route_resolver = object()

    def build(**_kwargs):
        if realtime:
            assert _kwargs["route_resolver"] is route_resolver
        else:
            assert "route_resolver" not in _kwargs
        seen.append(context_timing._SPANS.get() is not None and not gc.isenabled())
        raise RuntimeError("stop after the context build")

    monkeypatch.setattr(agent_module, "build_llm_messages", build)
    monkeypatch.setattr(provider_models, "model_has_credentials", lambda _model: True)
    monkeypatch.setattr(OuroborosAgent, "_log_worker_boot_once", lambda self: None)
    agent = OuroborosAgent(Env(repo_dir=pathlib.Path(__file__).resolve().parents[1], drive_root=tmp_path),
                           event_queue=queue.Queue())
    if realtime:
        agent._realtime_turn = SimpleNamespace(resolve_route=route_resolver)
    monkeypatch.setattr(agent.llm, "chat", lambda **_kw: pytest.fail("no model call"))
    with pytest.raises(RuntimeError, match="stop after the context build"):
        agent._prepare_task_context({"id": "timed", "type": "task", "chat_id": 1, "text": "hi"})
    assert seen == [True]
    assert context_timing._SPANS.get() is None and gc.isenabled()
