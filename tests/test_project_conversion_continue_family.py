"""Turning a Continue family's work into a project: lineage, destination ids, room joins and refusals.

Synthetic fixtures only. The naming model is a stub; nothing leaves the process.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from ouroboros import dialogue_store


def _request(tmp_path, body):
    async def _json():
        return body

    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(drive_root=tmp_path)), json=_json)


def _convert(tmp_path, **body):
    reply = asyncio.run(_post(tmp_path, body))
    return reply.status_code, json.loads(reply.body)


async def _post(tmp_path, body):
    from ouroboros.gateway.projects import api_project_from_task

    return await api_project_from_task(_request(tmp_path, body))


def _no_model_call(monkeypatch, why="no naming call is needed"):
    from ouroboros import project_naming

    async def forbidden(*_args, **_kwargs):
        pytest.fail(why)

    monkeypatch.setattr(project_naming, "llm_project_name_async", forbidden)


def _model(monkeypatch, name="Harbor map"):
    from ouroboros import project_naming

    calls = []

    async def coin(text, **_kwargs):
        calls.append(text)
        return name

    monkeypatch.setattr(project_naming, "llm_project_name_async", coin)
    return calls


def _family(tmp_path, monkeypatch, first="first", **fields):
    """An unnamed interrupted first root and its admitted Continue; returns (successor id, workers)."""
    from supervisor.continuation_admission import admit_continuation
    from tests._budget_pause_exact_helpers import _install_queue
    from tests._owner_continue_helpers import NONCE, _interrupted

    _q, _state, workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, first, **{"title": None, **fields})
    ack = admit_continuation(first, action_nonce=NONCE)
    assert ack["ok"], ack
    return ack["successor_task_id"], workers


def _patch(tmp_path, task_id, *, drop=(), **changes):
    from ouroboros.task_results import stamp_task_result_schema
    from ouroboros.task_store import patch_document

    patch_document(tmp_path, task_id, lambda current: stamp_task_result_schema(
        {**{k: v for k, v in current.items() if k not in drop}, **changes}))


def _origin(text, client_id="owner-start"):
    from ouroboros.project_dialogue import build_owner_message_ref

    return build_owner_message_ref(chat_id=1, client_message_id=client_id, ts="2026-06-18T00:00:00Z", text=text)


def _sentence_as_written(payload):
    """A refusal's 409 ``error`` is the one authoritative sentence the card shows: plain text, no markup."""
    from ouroboros.project_placement import NAMING_REFUSALS

    sentence = NAMING_REFUSALS[payload["reason_code"]]
    assert payload["error"] == sentence and "<" not in sentence and sentence.endswith(".")


def test_a_stale_claim_mark_does_not_hide_a_durably_admitted_family(tmp_path, monkeypatch):
    from ouroboros.projects_registry import project_binding_for_task
    from supervisor.continuation_admission import verified_work_episodes

    text = "Draw the coastline near the old harbor."
    successor, _workers = _family(tmp_path, monkeypatch, origin_message_ref=_origin(text), origin_message_text=text)
    claim = json.loads(json.dumps(__import__("ouroboros.task_results", fromlist=["x"]).load_task_result(
        tmp_path, "first")["continued_by"]))
    assert claim["state"] == "admitted"
    _patch(tmp_path, "first", continued_by={**claim, "state": "bound"})  # the best-effort mark never landed
    assert verified_work_episodes(tmp_path, successor) == ("first", successor)
    assert verified_work_episodes(tmp_path, "first") == ("first", successor)
    calls = _model(monkeypatch, "Old harbor map")
    status, payload = _convert(tmp_path, task_id="first")
    assert status == 200 and calls == [text], payload
    status, joined = _convert(tmp_path, task_id=successor)
    assert status == 200 and joined["adopted"] is True and calls == [text], joined
    assert project_binding_for_task(tmp_path, successor)["project_id"] == payload["project"]["id"]


def test_a_recorded_admission_with_an_unconfirmed_final_publication_is_pending_until_the_same_nonce_replays(
        tmp_path, monkeypatch):
    from supervisor.continuation_admission import admit_continuation, verified_work_episodes
    from tests._budget_pause_exact_helpers import _install_queue
    from tests._owner_continue_helpers import NONCE, _interrupted

    text = "Draw the coastline near the old harbor."
    queue, _state, _workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, "first", title=None, origin_message_ref=_origin(text), origin_message_text=text)
    real = queue.persist_queue_snapshot
    monkeypatch.setattr(queue, "persist_queue_snapshot",
                        lambda reason="": False if reason == "owner_continue_committed" else real(reason=reason))
    failed = admit_continuation("first", action_nonce=NONCE)  # the result row is written, the final publication is not
    assert failed["ok"] is False and failed["error"] == "queue_snapshot_persist_failed"
    successor = failed["successor_task_id"]
    _no_model_call(monkeypatch, "an unconfirmed Continue is neither standalone nor named from its wrapper")
    for clicked in (successor, "first"):
        with pytest.raises(ValueError) as pending:
            verified_work_episodes(tmp_path, clicked)
        assert getattr(pending.value, "reason", "") == "continuation_pending"
    status, payload = _convert(tmp_path, task_id=successor)
    assert status == 409 and payload["reason_code"] == "continuation_pending", payload
    monkeypatch.setattr(queue, "persist_queue_snapshot", real)
    replay = admit_continuation("first", action_nonce=NONCE)  # the same press confirms the one admission
    assert replay["ok"] is True and replay["successor_task_id"] == successor
    assert verified_work_episodes(tmp_path, successor) == ("first", successor)
    assert verified_work_episodes(tmp_path, "first") == ("first", successor)


def _pending_during(queue, monkeypatch, when, clicks, seen):
    """Wrap the publication barrier of ``when``: once the live row's marker has left memory, before the document
    confirms it, each clicked episode is read and what it answers is appended to ``seen``."""
    from supervisor.continuation_admission import verified_work_episodes

    real = queue.publish_queue_snapshot

    def publish(reason=""):
        if reason == when:
            assert queue.PENDING and not any(row.get("_continuation_prepared") for row in queue.PENDING)
            for clicked in clicks(queue):
                try:
                    seen.append((clicked, verified_work_episodes(queue.DRIVE_ROOT, clicked)))
                except ValueError as exc:
                    seen.append((clicked, getattr(exc, "reason", "")))
        return real(reason=reason)

    monkeypatch.setattr(queue, "publish_queue_snapshot", publish)


def test_a_family_is_admitted_only_when_the_published_queue_confirms_it(tmp_path, monkeypatch):
    """First admission: the marker leaves the live row before the last publication confirms, so memory is ahead."""
    from supervisor.continuation_admission import admit_continuation, verified_work_episodes
    from tests._budget_pause_exact_helpers import _install_queue
    from tests._owner_continue_helpers import NONCE, _interrupted

    text = "Draw the coastline near the old harbor."
    queue, _state, _workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, "first", title=None, origin_message_ref=_origin(text), origin_message_text=text)
    window = []
    _pending_during(queue, monkeypatch, "owner_continue_committed",
                    lambda q: ["first", *(str(row["id"]) for row in q.PENDING)], window)
    ack = admit_continuation("first", action_nonce=NONCE)
    assert ack["ok"], ack
    successor = ack["successor_task_id"]
    assert window == [("first", "continuation_pending"), (successor, "continuation_pending")]
    for clicked in ("first", successor):
        assert verified_work_episodes(tmp_path, clicked) == ("first", successor)


def test_a_same_nonce_replay_is_pending_until_its_own_publication_confirms(tmp_path, monkeypatch):
    from supervisor.continuation_admission import admit_continuation, verified_work_episodes
    from tests._budget_pause_exact_helpers import _install_queue
    from tests._owner_continue_helpers import NONCE, _interrupted

    text = "Draw the coastline near the old harbor."
    queue, _state, _workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, "first", title=None, origin_message_ref=_origin(text), origin_message_text=text)
    persist = queue.persist_queue_snapshot
    monkeypatch.setattr(queue, "persist_queue_snapshot",
                        lambda reason="": False if reason == "owner_continue_committed" else persist(reason=reason))
    failed = admit_continuation("first", action_nonce=NONCE)  # the result row is written, the last publication is not
    assert failed["ok"] is False and failed["error"] == "queue_snapshot_persist_failed"
    successor = failed["successor_task_id"]
    monkeypatch.setattr(queue, "persist_queue_snapshot", persist)
    window = []
    _pending_during(queue, monkeypatch, "owner_continue_binding_recovered", lambda q: ["first", successor], window)
    replay = admit_continuation("first", action_nonce=NONCE)
    assert replay["ok"] is True and replay["replay"] is True and replay["successor_task_id"] == successor
    assert window == [("first", "continuation_pending"), (successor, "continuation_pending")]
    for clicked in ("first", successor):
        assert verified_work_episodes(tmp_path, clicked) == ("first", successor)


def test_a_confirmed_admission_stands_beside_a_predecessor_mark_that_never_landed(tmp_path, monkeypatch):
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import verified_work_episodes

    text = "Draw the coastline near the old harbor."
    successor, _workers = _family(tmp_path, monkeypatch, origin_message_ref=_origin(text), origin_message_text=text)
    _patch(tmp_path, "first", continued_by={**load_task_result(tmp_path, "first")["continued_by"], "state": "bound"})
    for clicked in ("first", successor):
        assert verified_work_episodes(tmp_path, clicked) == ("first", successor)
    _no_model_call(monkeypatch, "a confirmed family's first episode is named from its own opening")
    calls = _model(monkeypatch, "Old harbor map")
    status, payload = _convert(tmp_path, task_id=successor)
    assert status == 200 and calls == [text], payload


def test_a_trusted_claim_names_an_api_continue_whose_receipt_is_missing_but_caller_metadata_names_nothing(
        tmp_path, monkeypatch):
    from ouroboros.task_results import write_task_result

    text = "Draw the coastline near the old harbor."
    successor, _workers = _family(tmp_path, monkeypatch, origin_message_ref=_origin(text), origin_message_text=text)
    receipt = {"status": "accepted", "token": "admission"}
    _patch(tmp_path, successor, drop=("continuation_admission",), api_admission=receipt)  # the host receipt is missing
    _no_model_call(monkeypatch, "a Continue its predecessor's own claim names is no origin")
    status, payload = _convert(tmp_path, task_id=successor)
    assert status == 409 and payload["reason_code"] == "continuation_pending", payload
    # A root whose caller wrote Continue metadata naming a predecessor that claimed another successor stays an origin.
    write_task_result(tmp_path, "api-root", "completed", description="Count the street trees.", api_admission=receipt,
                      metadata={"client_surface": {"channel": "api_task"}, "continuation": {"predecessor_task_id": "first"}})
    calls = _model(monkeypatch, "Street tree count")
    status, root = _convert(tmp_path, task_id="api-root")
    assert status == 200 and root["project"]["name"] == "Street tree count", root
    assert calls == [json.dumps({"description": "Count the street trees."})]


def test_a_continue_whose_admission_is_not_recorded_is_pending_not_standalone(tmp_path, monkeypatch):
    from ouroboros.projects_registry import get_project, project_binding_for_task

    text = "Draw the coastline near the old harbor."
    successor, _workers = _family(tmp_path, monkeypatch, origin_message_ref=_origin(text), origin_message_text=text)
    _patch(tmp_path, successor, drop=("continuation_admission",))  # the result row exists, its admission record does not
    _no_model_call(monkeypatch, "a pending Continue's wrapper label is not a subject")
    status, payload = _convert(tmp_path, task_id=successor)
    assert status == 409 and payload["reason_code"] == "continuation_pending", payload
    _sentence_as_written(payload)
    assert "stayed in Main" in payload["error"] and "again in a moment" in payload["error"]
    assert project_binding_for_task(tmp_path, successor) is None and get_project(tmp_path, f"task-{successor}"[:64]) is None
    # An explicit name is the owner's own choice: it never waits for lineage.
    status, named = _convert(tmp_path, task_id=successor, name="Harbor coast")
    assert status == 200 and named["project"]["name"] == "Harbor coast"


def test_an_authored_title_that_begins_with_continue_is_the_name_and_a_wrapper_never_is(tmp_path, monkeypatch):
    from ouroboros.task_results import write_task_result

    # An ordinary root whose owner-authored title happens to start with the word: it is that title, no model call.
    write_task_result(tmp_path, "authored", "completed", title="Continue the breakwater survey")
    _no_model_call(monkeypatch)
    status, payload = _convert(tmp_path, task_id="authored")
    assert status == 200 and payload["project"]["name"] == "Continue the breakwater survey", payload
    write_task_result(tmp_path, "colon", "completed", title="Continue: the harbor survey")  # a human's own title
    status, payload = _convert(tmp_path, task_id="colon")
    assert status == 200 and payload["project"]["name"] == "Continue: the harbor survey", payload
    # A real Continue's wrapper ("Continue: ...") is not read: the family is named from its first episode's opening.
    text = "Survey the northern breakwater."
    successor, _workers = _family(tmp_path, monkeypatch, "root-a", origin_message_ref=_origin(text), origin_message_text=text)
    calls = _model(monkeypatch, "Breakwater survey")
    status, payload = _convert(tmp_path, task_id=successor)
    assert status == 200 and calls == [text] and payload["project"]["name"] == "Breakwater survey", payload


def test_a_family_past_the_verification_bound_is_refused_with_its_own_reason(tmp_path, monkeypatch):
    import supervisor.continuation_admission as admission

    text = "Draw the coastline near the old harbor."
    successor, _workers = _family(tmp_path, monkeypatch, origin_message_ref=_origin(text), origin_message_text=text)
    monkeypatch.setattr(admission, "_WORK_CHAIN_MAX_HOPS", 1)
    _no_model_call(monkeypatch)
    status, payload = _convert(tmp_path, task_id=successor)
    assert status == 409 and payload["reason_code"] == "continuation_family_too_long", payload
    _sentence_as_written(payload)
    status, named = _convert(tmp_path, task_id=successor, name="Harbor coast")
    assert status == 200, named


def test_a_family_of_exactly_the_bound_is_not_too_long(tmp_path, monkeypatch):
    import supervisor.continuation_admission as admission

    text = "Draw the coastline near the old harbor."
    successor, _workers = _family(tmp_path, monkeypatch, origin_message_ref=_origin(text), origin_message_text=text)
    monkeypatch.setattr(admission, "_WORK_CHAIN_MAX_HOPS", 2)
    assert admission.verified_work_episodes(tmp_path, "first") == ("first", successor)
    assert admission.verified_work_episodes(tmp_path, successor) == ("first", successor)
    monkeypatch.setattr(admission, "_WORK_CHAIN_MAX_HOPS", 1)
    with pytest.raises(admission.ContinueRefusal):
        admission.verified_work_episodes(tmp_path, "first")


def test_long_episode_ids_share_one_default_room_and_explicit_ids_are_kept(tmp_path, monkeypatch):
    from ouroboros.project_facts import explicit_project_id_ok
    from ouroboros.project_placement import conversion_destination
    from ouroboros.projects_registry import list_sidebar_projects

    long_id = "long-" + "a" * 43  # 48 characters: its successor id is 66, and ``task-`` + that exceeds a project id
    text = "Draw the coastline near the old harbor."
    successor, _workers = _family(tmp_path, monkeypatch, first=long_id,
                                  origin_message_ref=_origin(text), origin_message_text=text)
    assert len(successor) == 66
    short, short_default = conversion_destination("first", "")
    assert (short, short_default) == ("task-first", True), "the established short default is unchanged"
    derived, derived_default = conversion_destination(successor, "")
    assert derived_default and derived != f"task-{successor}"[:64] and explicit_project_id_ok(derived)
    assert conversion_destination(successor, "")[0] == derived, "the derived id is stable"
    assert conversion_destination("first", "my-room") == ("my-room", False)
    older = f"task-{successor}"[:64]  # what an older browser sent, with no name, for a long task id
    assert conversion_destination(successor, older) == (derived, True), "the truncated id gives way to the whole-id default"
    assert conversion_destination(successor, older, named=True) == (older, False), "beside a name it is the caller's own id"
    calls = _model(monkeypatch, "Old harbor map")
    status, first = _convert(tmp_path, task_id=long_id)  # nothing on the wire but the task
    assert status == 200 and first["project"]["name"] == "Old harbor map", first
    status, joined = _convert(tmp_path, task_id=successor)
    assert status == 200 and joined["adopted"] is True and joined["project"]["id"] == first["project"]["id"], joined
    status, again = _convert(tmp_path, task_id=successor, id=older)
    assert status == 200 and again["project"]["id"] == first["project"]["id"] and calls == [text], again
    assert len(list_sidebar_projects(tmp_path)) == 1


def _two_long_works(tmp_path, text="Sketch the lighthouse."):
    """Two independent works whose long ids agree up to the 64 characters an older browser's id was cut to."""
    from ouroboros.task_results import write_task_result

    north, south = f"{'n' * 60}-north", f"{'n' * 60}-south"
    older = f"task-{north}"[:64]
    assert older == f"task-{south}"[:64]
    for tid in (north, south):
        write_task_result(tmp_path, tid, "completed", origin_message_ref=_origin(text, tid), origin_message_text=text)
    return north, south, older


def test_the_server_default_serves_the_canonical_and_the_deprecated_unnamed_request(tmp_path, monkeypatch):
    from ouroboros.project_placement import conversion_destination
    from ouroboros.projects_registry import list_sidebar_projects, project_binding_for_task
    from ouroboros.task_results import write_task_result

    north, south, older = _two_long_works(tmp_path)
    for tid in ("short-a", "short-b"):
        write_task_result(tmp_path, tid, "completed", origin_message_ref=_origin("Sketch the pier.", tid),
                          origin_message_text="Sketch the pier.")
    _model(monkeypatch, "Lighthouse sketch")
    status, modern = _convert(tmp_path, task_id=north)  # the canonical request: the task and nothing else
    assert status == 200 and modern["project"]["id"] == conversion_destination(north)[0] != older, modern
    status, legacy = _convert(tmp_path, task_id=south, id=older, name="")  # the older unnamed shape
    assert status == 200 and legacy["project"]["id"] == conversion_destination(south)[0], legacy
    assert legacy["project"]["id"] not in {modern["project"]["id"], older}
    assert project_binding_for_task(tmp_path, south)["project_id"] == legacy["project"]["id"]
    status, short_modern = _convert(tmp_path, task_id="short-a")  # short ids keep the established default
    assert status == 200 and short_modern["project"]["id"] == "task-short-a", short_modern
    status, short_legacy = _convert(tmp_path, task_id="short-b", id="task-short-b")
    assert status == 200 and short_legacy["project"]["id"] == "task-short-b", short_legacy
    assert len(list_sidebar_projects(tmp_path)) == 4


def test_an_unrelated_work_never_joins_a_room_on_the_strength_of_the_older_truncated_id(tmp_path, monkeypatch):
    from ouroboros.projects_registry import get_project, list_sidebar_projects, project_binding_for_task

    north, south, older = _two_long_works(tmp_path)
    _model(monkeypatch, "Lighthouse sketch")
    status, own = _convert(tmp_path, task_id=north, id=older, name="Northern room")  # beside a name: the caller's id
    assert status == 200 and own["project"]["id"] == older, own
    status, other = _convert(tmp_path, task_id=south, id=older)  # an unbound, unrelated work, the older unnamed shape
    assert status == 200 and other["adopted"] is False and other["project"]["id"] != older, other
    assert other["project"]["name"] == "Lighthouse sketch" and get_project(tmp_path, older)["name"] == "Northern room"
    assert project_binding_for_task(tmp_path, south)["project_id"] == other["project"]["id"]
    for body in ({}, {"id": older}):  # a bound work keeps its exact binding whichever shape repeats the call
        status, again = _convert(tmp_path, task_id=north, **body)
        assert status == 200 and again["adopted"] is True and again["project"]["id"] == older, again
    assert len(list_sidebar_projects(tmp_path)) == 2


def test_an_explicit_destination_and_its_registered_name_are_not_rewritten(tmp_path, monkeypatch):
    from ouroboros.task_results import write_task_result

    text = "Sketch the lighthouse."
    for tid in ("keep-a", "keep-b"):
        write_task_result(tmp_path, tid, "completed", origin_message_ref=_origin(text, tid), origin_message_text=text)
    _model(monkeypatch, "Lighthouse sketch")
    status, auto = _convert(tmp_path, task_id="keep-a", id="lighthouse")
    assert status == 200 and auto["project"]["id"] == "lighthouse" and auto["project"]["name"] == "Lighthouse sketch"
    status, named = _convert(tmp_path, task_id="keep-a", id="lighthouse", name="Beacon")  # the bound work's own room
    assert status == 200 and named["project"]["id"] == "lighthouse" and named["project"]["name"] == "Beacon", named
    status, other = _convert(tmp_path, task_id="keep-b", id="lighthouse", name="Ignored")  # another work joins, never renames
    assert status == 200 and other["project"]["id"] == "lighthouse" and other["project"]["name"] == "Beacon", other
    status, bad = _convert(tmp_path, task_id="keep-b", id="Not Valid")
    assert status == 400, bad


def test_an_ordinary_legacy_opening_is_restored_by_its_exact_hash_verified_ref(tmp_path, monkeypatch):
    from ouroboros.task_results import write_task_result

    text = "Map the two tributaries and label their source."
    row = {"chat_id": 1, "client_message_id": "legacy-1", "ts": "2026-06-18T00:00:00Z", "text": text}
    dialogue_store.append(tmp_path, "chat", {**row, "direction": "in"})
    write_task_result(tmp_path, "legacy", "completed", origin_message_ref=_origin(text, "legacy-1"))  # no text copy
    calls = _model(monkeypatch, "Tributary map")
    status, payload = _convert(tmp_path, task_id="legacy")
    assert status == 200 and calls == [text], payload
    assert payload["binding"]["source_text"] == text
    # A ref whose row is gone (or whose words changed) is a genuinely missing source: the narrower refusal.
    write_task_result(tmp_path, "gone", "completed", origin_message_ref=_origin("Words that were never stored", "gone"))
    _no_model_call(monkeypatch, "nothing recoverable may be sent")
    status, refused = _convert(tmp_path, task_id="gone")
    assert status == 409 and refused["reason_code"] == "project_name_source_unavailable", refused
    _sentence_as_written(refused)
    assert refused["error"] == "This work couldn't be named automatically, so it stayed in Main."


def _chat_title_message(content, **extra):
    return {"role": "assistant", "content": content, **extra}


@pytest.mark.parametrize("message, usage, expected", [
    (_chat_title_message("Harbor map"), {}, "Harbor map"),
    (_chat_title_message([{"type": "text", "text": "Harbor "}, {"type": "text", "text": "map"}]), {}, "Harbor map"),
    (_chat_title_message([{"type": "thinking", "text": "Let me think"}, {"type": "text", "text": "Harbor map"}]), {}, "Harbor map"),
    (_chat_title_message([{"type": "reasoning", "text": "Only thoughts"}]), {}, ""),
    # Gemini parts carry no type: a truthy ``thought`` is the whole marker, whatever its spelling.
    (_chat_title_message([{"text": "I should pick a short name.", "thought": True}, {"text": "Harbor map"}]), {}, "Harbor map"),
    (_chat_title_message([{"text": "Weighing two names.", "thought": "true"}, {"text": "Harbor map"}]), {}, "Harbor map"),
    (_chat_title_message([{"text": "Only a thought.", "thought": True}]), {}, ""),
    (_chat_title_message([{"text": "Harbor map", "thought": False}]), {}, "Harbor map"),
    (_chat_title_message("", refusal="I can't help with that."), {}, ""),
    (_chat_title_message("I can't help with that.", refusal="I can't help with that."), {}, ""),
    (_chat_title_message("I can't help with that.", finish_reason="content_filter"), {}, ""),
    (_chat_title_message("I can't help with that."), {"response_finish_reason": "refusal"}, ""),
    (_chat_title_message("I can't help with that.", stop_reason="refusal"), {}, ""),
    (_chat_title_message("Half a tit"), {"response_finish_reason": "length"}, ""),
    (_chat_title_message("Harbor map"), {"response_status": "incomplete", "response_finish_reason": "max_output_tokens"}, ""),
    (_chat_title_message("   \n"), {}, ""),
    (None, {}, ""),
])
def test_only_visible_model_text_becomes_a_title(message, usage, expected):
    from ouroboros.project_naming import visible_model_title

    assert visible_model_title(message, usage) == expected


@pytest.mark.parametrize("reply, usage", [
    ({"role": "assistant", "content": "", "refusal": "No."}, {"response_finish_reason": "stop"}),
    ({"role": "assistant", "content": "No.", "finish_reason": "content_filter"}, {}),
    ({"role": "assistant", "content": [{"type": "reasoning", "text": "hidden"}]}, {}),
])
def test_the_shared_light_call_returns_no_title_for_a_refusal(tmp_path, reply, usage):
    from ouroboros import project_naming

    class Client:
        def chat(self, **_kwargs):
            return dict(reply), dict(usage)

    assert project_naming.llm_project_name("Name the harbor map", llm_client=Client(), single_attempt=True) == ""
    assert project_naming.llm_project_name("Name the harbor map", llm_client=Client()) == ""


def test_a_refusing_provider_leaves_the_work_in_main_through_the_gateway(tmp_path, monkeypatch):
    from ouroboros import project_naming
    from ouroboros.projects_registry import project_binding_for_task
    from ouroboros.task_results import write_task_result

    text = "Sketch the lighthouse."
    write_task_result(tmp_path, "refused", "completed", origin_message_ref=_origin(text, "refused"), origin_message_text=text)

    class Client:
        def chat(self, **_kwargs):
            return {"role": "assistant", "content": "I cannot title this.", "refusal": "I cannot title this."}, {}

    real = project_naming.llm_project_name
    monkeypatch.setattr(project_naming, "llm_project_name",
                        lambda *args, **kwargs: real(*args, **{**kwargs, "llm_client": Client()}))
    status, payload = _convert(tmp_path, task_id="refused")
    assert status == 409 and payload["reason_code"] == "project_name_unavailable", payload
    _sentence_as_written(payload)
    assert "Turn into project" in payload["error"], "a retry of the existing button, never a new chat"
    assert project_binding_for_task(tmp_path, "refused") is None


def test_a_pre_bound_sibling_is_joined_with_its_fresh_folder_and_without_a_name_call(tmp_path, monkeypatch):
    from ouroboros.projects_registry import project_binding_for_task, update_project

    text = "Draw the coastline near the old harbor."
    successor, _workers = _family(tmp_path, monkeypatch, origin_message_ref=_origin(text), origin_message_text=text)
    status, room = _convert(tmp_path, task_id="first", name="Harbor coast")
    assert status == 200
    new_folder = tmp_path / "moved-folder"
    new_folder.mkdir()
    update_project(tmp_path, room["project"]["id"], working_dir=str(new_folder))  # before the click
    _no_model_call(monkeypatch, "a bound sibling's room needs no name")
    status, joined = _convert(tmp_path, task_id=successor)
    assert status == 200 and joined["adopted"] is True, joined
    assert joined["project"]["id"] == room["project"]["id"] and joined["project"]["name"] == "Harbor coast"
    assert project_binding_for_task(tmp_path, successor)["placement_admission"]["project"]["working_dir"] == str(new_folder)


def test_a_sibling_bound_after_preparation_is_joined_under_a_fresh_fence_not_its_old_one(tmp_path, monkeypatch):
    from ouroboros.gateway.projects import _prepare_conversion
    from ouroboros.project_placement import move_task_to_project
    from ouroboros.projects_registry import project_binding_for_task, update_project

    text = "Draw the coastline near the old harbor."
    successor, workers = _family(tmp_path, monkeypatch, origin_message_ref=_origin(text), origin_message_text=text)
    default = f"task-{successor}"
    origin, basis, _name, _reason, _text = _prepare_conversion(tmp_path, successor, default, "", True)
    assert basis["project"] is None  # prepared before any sibling was bound
    status, room = _convert(tmp_path, task_id="first", name="Harbor coast")
    assert status == 200
    old_fence = project_binding_for_task(tmp_path, "first")["placement_admission"]["project"]["working_dir"]
    new_folder = tmp_path / "folder-after-the-sibling-bound"
    new_folder.mkdir()
    update_project(tmp_path, room["project"]["id"], working_dir=str(new_folder))
    outcome = move_task_to_project(tmp_path, successor, default, origin=origin, basis=basis, running=workers.RUNNING,
                                   pending=workers.PENDING, initiated_by="owner", automatic_name=True,
                                   adopt_work_room=True)
    assert outcome["adopted"] is True and outcome["project_id"] == room["project"]["id"]
    admitted = project_binding_for_task(tmp_path, successor)["placement_admission"]["project"]["working_dir"]
    assert admitted == str(new_folder) != old_fence


def test_a_clicked_episode_keeps_its_own_binding_over_a_sibling(tmp_path, monkeypatch):
    text = "Draw the coastline near the old harbor."
    successor, _workers = _family(tmp_path, monkeypatch, origin_message_ref=_origin(text), origin_message_text=text)
    assert _convert(tmp_path, task_id="first", name="Harbor coast")[0] == 200
    status, own = _convert(tmp_path, task_id=successor, id="own-room", name="Own room")
    assert status == 200 and own["project"]["id"] == "own-room"
    _no_model_call(monkeypatch)
    status, again = _convert(tmp_path, task_id=successor)
    assert status == 200 and again["project"]["id"] == "own-room" and again["adopted"] is True, again


@pytest.mark.parametrize("stage", ["deleting", "tombstoned"])
def test_a_deleted_sibling_is_neither_adopted_nor_resurrected(tmp_path, monkeypatch, stage):
    from ouroboros.projects_registry import begin_project_deletion, get_reserved_project, project_binding_for_task

    text = "Draw the coastline near the old harbor."
    successor, _workers = _family(tmp_path, monkeypatch, origin_message_ref=_origin(text), origin_message_text=text)
    status, room = _convert(tmp_path, task_id="first", name="Harbor coast")
    assert status == 200
    room_id = room["project"]["id"]
    begin_project_deletion(tmp_path, room_id)
    if stage == "tombstoned":
        from ouroboros.projects_registry import complete_project_deletion

        complete_project_deletion(tmp_path, room_id)
    before = get_reserved_project(tmp_path, room_id)
    _no_model_call(monkeypatch, "a deleted room is no destination to name")
    status, payload = _convert(tmp_path, task_id=successor)
    assert status == 409 and payload["reason_code"] == "project_routing_fence", payload
    assert project_binding_for_task(tmp_path, successor) is None
    assert get_reserved_project(tmp_path, room_id) == before, "the fence or tombstone is untouched"
