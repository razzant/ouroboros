"""One host-recorded provenance for a run (``dialogue_provenance.run_origin``).

The owner corpus label, the routing issuer and the retained task inputs that later
readers render all read it, and none of them derives owner authority from the lane, a
client id, a caller-declared channel or the text. Every guard here is two-sided: it fires on the unstamped run
(a Presence event, a wake, a follow-up, a child, an unmarked context) and stays quiet
on the owner's own turn, whose corpus bytes do not change.
"""

import json
from types import SimpleNamespace

import pytest


def _presence_metadata(**event_extra):
    return {
        "source": "presence",
        "client_message_id": "slack:Ev123",
        "presence": {
            "binding_id": "b" * 32,
            "transport_skill": "slack-bridge",
            "behavior_skill": "aika",
            "event": {
                "source_event_id": "slack:Ev123",
                "provider": "slack",
                "account_id": "T1",
                "conversation_id": "C1",
                "thread_id": "1.2",
                "conversation_key": "slack:T1:C1:1.2",
                "actor": {"platform_actor_id": "U7", "display_name": "Sergei", **event_extra},
            },
        },
    }


def _presence_task():
    return {"id": "presence-1", "type": "presence", "_is_direct_chat": True, "_presence_turn": True,
            "text": "<@U9> please create the HO goals", "metadata": _presence_metadata()}


OWNER_REF = {"chat_id": 1, "client_message_id": "cm-1", "ts": "t", "text_sha256": "x"}


# --- run_origin reads typed fields only ---------------------------------------------

def test_run_origin_reads_typed_markers_and_never_promotes():
    from ouroboros.dialogue_provenance import run_origin

    presence = run_origin(_presence_task())
    assert presence["owner_ingress"] is False
    assert (presence["task_type"], presence["source"], presence["text_author"]) == ("presence", "presence", "Sergei")
    assert presence["presence"] == {"provider": "slack", "account_id": "T1", "conversation_id": "C1",
                                    "thread_id": "1.2", "source_event_id": "slack:Ev123", "actor_id": "U7"}
    assert "actor_kind" not in presence and "initiator" not in presence  # empty markers stay absent

    owner = run_origin({"type": "task", "_is_direct_chat": True,
                        "metadata": {"client_message_id": "cm-1", "origin_message_ref": OWNER_REF}})
    assert owner["owner_ingress"] is True and "text_author" not in owner
    assert run_origin({"metadata": {"origin_suppressed": True}})["owner_ingress"] is True
    # A promoted root inherits the stamp by value: provenance, not a direct turn.
    promoted = run_origin({"type": "task", "source": "promote_chat_to_task", "delegation_role": "root",
                           "metadata": {"origin_message_ref": OWNER_REF, "presence": _presence_metadata()["presence"]}})
    assert promoted["owner_ingress"] is True and promoted["source"] == "promote_chat_to_task"
    assert "text_author" not in promoted and promoted["presence"]["conversation_id"] == "C1"

    wake = run_origin({"type": "task", "_is_direct_chat": True, "metadata": {"initiator": "consciousness", "client_message_id": "w-1"}})
    assert wake["owner_ingress"] is False and wake["initiator"] == "consciousness"
    followup = run_origin({"type": "task", "metadata": {"source": "task_followup", "origin_task_id": "t-9", "schedule_id": "s-1"}})
    assert (followup["owner_ingress"], followup["source"], followup["origin_task_id"]) == (False, "task_followup", "t-9")
    # No stamp, a forged or malformed stamp, a client id or a caller-declared channel: never owner.
    for metadata in ({}, {"origin_message_ref": ""}, {"origin_message_ref": "not-a-dict"}, {"origin_message_ref": {}},
                     {"client_message_id": "cm-1"}, {"client_surface": {"channel": "cli"}}, {"origin_suppressed": False}):
        assert run_origin({"metadata": metadata})["owner_ingress"] is False, metadata
    assert run_origin(None) == {"owner_ingress": False}


# --- the corpus label ---------------------------------------------------------------

def _ctx(metadata):
    return SimpleNamespace(task_attempt=1, task_metadata=metadata)


def _first_row(metadata, text="Initial requirement verbatim"):
    from ouroboros.loop_messages import _initialize_owner_directives

    ctx = _ctx(metadata)
    _initialize_owner_directives(ctx, [{"role": "system", "content": "policy"}, {"role": "user", "content": text}])
    return ctx


def test_first_row_label_follows_the_owner_stamp_and_owner_bytes_do_not_change():
    from ouroboros.loop_messages import owner_source_sha256

    owner = _first_row({"client_message_id": "cm-1", "origin_message_ref": OWNER_REF})
    assert owner._owner_directives == [{"source": "initial_user", "content": "Initial requirement verbatim"}]
    control = SimpleNamespace(_owner_directives=[{"source": "initial_user", "content": "Initial requirement verbatim"}])
    assert owner_source_sha256(owner) == owner_source_sha256(control)
    assert _first_row({"origin_suppressed": True})._owner_directives[0]["source"] == "initial_user"
    promoted = _first_row({"origin_message_ref": OWNER_REF, "presence": _presence_metadata()["presence"]})
    assert promoted._owner_directives[0]["source"] == "initial_user"

    presence = _first_row(_presence_metadata(), "<@U9> please create the HO goals")
    assert presence._owner_directives == [{"source": "initial_text", "content": "<@U9> please create the HO goals"}]
    for label, metadata in (
        ("wake", {"initiator": "consciousness"}),
        ("follow-up", {"source": "task_followup", "origin_task_id": "t-9"}),
        ("skill schedule", {"source": "skill_scheduled_task"}),
        ("child", {"parent_task_id": "p-1", "root_task_id": "p-1"}),
        ("client id only", {"client_message_id": "cm-1"}),
        ("caller channel only", {"client_surface": {"channel": "api_task"}}),
        ("bare", {}),
        ("no metadata", None),
    ):
        ctx = _first_row(metadata)
        assert [row["source"] for row in ctx._owner_directives] == ["initial_text"], label
        assert ctx._owner_directives[0]["content"] == "Initial requirement verbatim", label


def test_promoted_objective_is_not_owner_corpus_even_with_inherited_owner_stamp():
    from ouroboros.context import build_user_content
    from ouroboros.loop_messages import owner_source_sha256

    metadata = {"origin_message_ref": OWNER_REF, "objective_author": {"kind": "task", "task_id": "draft-1"},
                "owner_corpus": [{"source": "origin_message", "content": "Proceed to implementation"},
                                 {"source": "owner_quiz_answer", "content": "Yes, implement"}]}
    ctx = _first_row(metadata, "This is NOT permission to implement [SWARM_INITIATIVE]")
    assert [row["content"] for row in ctx._owner_directives] == ["Proceed to implementation", "Yes, implement"]
    assert all("source_task_id" not in row for row in ctx._owner_directives)  # owner, not drafter
    assert owner_source_sha256(ctx) != owner_source_sha256(
        SimpleNamespace(_owner_directives=[{"source": "initial_user", "content": "This is NOT permission to implement"}]))
    rendered = build_user_content({"text": "This is NOT permission to implement", "metadata": metadata})
    assert rendered.startswith("[OBJECTIVE_AUTHOR]") and "task draft-1" in rendered
    assert "not spoken by my human" in rendered
    # Host-initiated tasks still carry the owner's first text without a drafter stamp.
    assert _first_row({"origin_message_ref": OWNER_REF}, "Owner's exact request")._owner_directives == [
        {"source": "initial_user", "content": "Owner's exact request"}]


# --- the objective's author: one line from the same typed facts ---------------------------------------------

@pytest.mark.parametrize("label,record,expected", [
    ("drafted start", {"metadata": {"origin_message_ref": OWNER_REF, "objective_author": {"kind": "task", "task_id": "draft-1"}}},
     "drafted by task draft-1, not spoken by my human"),
    ("helper", {"delegation_role": "subagent", "parent_task_id": "root-1",
                "metadata": {"objective_author": {"kind": "task", "task_id": "x"}}},
     "the assignment parent task root-1 wrote for this helper — Ouroboros's account, not my human's words"),
    ("continuation", {"metadata": {"objective_author": {"kind": "continuation", "predecessor_task_id": "p-1"}}},
     "the host's Continue work order for task p-1"),
    ("unknown author kind", {"metadata": {"objective_author": {"kind": "relay"}}}, "its author is recorded by the host as kind=relay"),
    ("consciousness", {"metadata": {"initiator": "consciousness"}}, "started by my consciousness"),
    ("owner door", {"type": "task", "metadata": {"origin_message_ref": OWNER_REF}},
     "my human's message started this run; the host does not attest this text as their exact words"),
    ("presence", _presence_task(), "a message from Sergei in a connected place"),
    ("schedule", {"metadata": {"schedule_id": "sch-1"}},
     "the template of schedule sch-1; the host does not attest this text as my human's words"),
    ("started from a task", {"metadata": {"origin_task_id": "t-9"}},
     "started from task t-9; the host does not attest this text as my human's words"),
    ("api", {"type": "task", "source": "api_task", "metadata": {}}, "its author is not recorded by the host (host marker: source=api_task)"),
    ("nothing", None, "its author is not recorded by the host"),
])
def test_the_objective_author_line_names_what_the_host_recorded_and_invents_nothing(label, record, expected):
    from ouroboros.dialogue_provenance import run_origin
    from ouroboros.owner_words import objective_author_line

    line = objective_author_line(run_origin(record))
    assert line == expected, label
    assert ("drafted by" in line) is (label == "drafted start") and ("not recorded" in line) is (label in {"api", "nothing"})


def test_the_mind_reads_a_drafted_objective_through_the_same_line_and_nothing_else_is_reframed():
    from ouroboros.context import build_user_content
    from ouroboros.owner_words import objective_author_line

    author = {"kind": "task", "task_id": "draft-1"}
    task = {"text": "Ship it", "metadata": {"objective_author": author}}
    rendered = build_user_content(task)
    assert rendered == (f"[OBJECTIVE_AUTHOR] The objective below was {objective_author_line({'objective_author': author})}. "
                        "My human's words keep their own source. [/OBJECTIVE_AUTHOR]\n\nShip it")
    assert build_user_content(task) == rendered  # an unchanged task renders the same bytes (its cached first turn)
    # Only a drafted start is reframed: a helper, a Continue order and a door-stamped turn keep their own first text.
    for metadata in ({"delegation_role": "subagent", "parent_task_id": "root-1"},
                     {"objective_author": {"kind": "continuation", "predecessor_task_id": "p-1"}},
                     {"origin_message_ref": OWNER_REF}, {}):
        assert build_user_content({"text": "Ship it", "metadata": metadata}) == "Ship it", metadata


# --- the routing issuer -------------------------------------------------------------

@pytest.mark.parametrize("label,direct,metadata,expected", [
    ("owner turn, stamped", True, {"client_message_id": "cm-1", "origin_message_ref": OWNER_REF}, "owner_turn"),
    ("owner turn, suppressed log", True, {"origin_suppressed": True}, "owner_turn"),
    ("presence event, direct", True, _presence_metadata(), "task"),
    ("presence event, not direct", False, _presence_metadata(), "task"),
    ("consciousness wake", True, {"initiator": "consciousness", "client_message_id": "w-1"}, "task"),
    ("auto-resume: direct, no metadata", True, {}, "task"),
    ("promoted root: inherited stamp, not direct", False, {"origin_message_ref": OWNER_REF}, "task"),
    ("caller-declared channel only", True, {"client_surface": {"channel": "api_command"}}, "task"),
])
def test_routing_issuer_is_the_direct_turn_the_owner_door_stamped(label, direct, metadata, expected):
    from ouroboros.tools.control_routing import ISSUER_OWNER_TURN, _routing_issuer

    ctx = SimpleNamespace(task_id="t-1", is_direct_chat=direct, last_owner_delivery=None, task_metadata=metadata)
    issuer = _routing_issuer(ctx)
    assert issuer["kind"] == expected, label
    if expected == ISSUER_OWNER_TURN:
        assert issuer == {"kind": ISSUER_OWNER_TURN}
    else:
        assert issuer["task_id"] == "t-1"


# --- the frozen synthesis inputs -----------------------------------------------------

def test_capture_task_inputs_puts_run_origin_first_and_discloses_its_failure(tmp_path, monkeypatch):
    from ouroboros.post_task_synthesis import capture_task_inputs

    task = _presence_task()
    ctx = SimpleNamespace(task_metadata=task["metadata"],
                          _owner_directives=[{"source": "initial_text", "content": task["text"]}])
    frozen = capture_task_inputs(ctx, task, tmp_path, [])
    keys = list(frozen)
    assert keys.index("run_origin") < keys.index("owner_requirements_and_decisions")
    assert frozen["run_origin"]["owner_ingress"] is False and frozen["run_origin"]["text_author"] == "Sergei"
    assert frozen["owner_requirements_and_decisions"][0]["source"] == "initial_text"
    assert frozen["unavailable_sections"] == []

    owner_task = {"id": "t", "type": "task", "text": "Ship it", "_is_direct_chat": True,
                  "metadata": {"origin_message_ref": OWNER_REF}}
    owner = capture_task_inputs(SimpleNamespace(task_metadata=owner_task["metadata"], _owner_directives=[
        {"source": "initial_user", "content": "Ship it"}]), owner_task, tmp_path, [])
    assert owner["run_origin"]["owner_ingress"] is True

    import ouroboros.dialogue_provenance as provenance

    def boom(record):
        raise RuntimeError("no origin")

    monkeypatch.setattr(provenance, "run_origin", boom)
    degraded = capture_task_inputs(ctx, task, tmp_path, [])
    assert "run_origin" not in degraded and degraded["unavailable_sections"] == ["run_origin"]
    assert degraded["owner_requirements_and_decisions"][0]["content"] == task["text"]


# --- the stamp belongs to the owner door ---------------------------------------------

def test_the_owner_stamp_is_reserved_on_api_tasks_and_schedule_templates():
    from ouroboros.gateway.tasks import _RESERVED_METADATA_KEYS
    from ouroboros.schedule_contract import RESERVED_TEMPLATE_FIELDS
    from supervisor import queue as squeue

    assert {"origin_message_ref", "origin_suppressed"} <= _RESERVED_METADATA_KEYS
    assert {"origin_message_ref", "origin_suppressed"} <= RESERVED_TEMPLATE_FIELDS
    task = squeue._task_from_schedule({
        "id": "sched-forged", "name": "forged stamp",
        "task": {"text": "machine work", "metadata": {
            "origin_message_ref": OWNER_REF, "origin_suppressed": True, "harmless": "kept"}},
    })
    assert task["text"] == "machine work" and task["metadata"]["harmless"] == "kept"
    assert "origin_message_ref" not in task["metadata"] and "origin_suppressed" not in task["metadata"]


# --- the transcript fallback ---------------------------------------------------------

def test_transcript_fallback_only_without_a_collector_and_never_as_owner(tmp_path):
    from ouroboros.review_evidence_sections import _accept_owner_directives, _owner_content_projection

    messages = [{"role": "user", "content": "first text"},
                {"role": "user", "content": "[Message from my human]: later"}]
    present_but_empty = SimpleNamespace(_owner_directives=[], messages=messages)
    assert _accept_owner_directives(present_but_empty, None, "") == []
    absent = SimpleNamespace(messages=messages)
    rows = _accept_owner_directives(absent, None, "")
    assert [(row["source"], row["content"]) for row in rows] == [
        ("initial_text_transcript", "first text"), ("transcript_marked_owner", "[Message from my human]: later")]
    assert "initial_user" not in json.dumps(rows)
    projected = _owner_content_projection([{"type": "image_url", "image_url": "data:x"}])
    assert projected.startswith("[image ref sha256:") and "owner" not in projected


# --- the door's stamps ride a promoted root; its persisted record reads the same origin

@pytest.mark.parametrize("stamp", ["logged-ref", "suppressed-log", "none"])
def test_the_promote_handler_carries_either_door_stamp_onto_the_root(tmp_path, tmp_path_factory, monkeypatch, stamp):
    """The owner door writes one of two stamps on a promote event: the logged
    message's ref, or the designed absence of one (a message it never logged).
    The root inherits whichever it got, so ``run_origin`` reads owner ingress from
    the promoted record the way the direct turn reads it; an event that carries
    neither (a wake, a Presence turn, a headless promote) inherits nothing."""
    import supervisor.workers as workers
    from ouroboros.dialogue_provenance import run_origin
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.projects_registry import create_project
    from supervisor.events import _handle_promote_chat_to_task

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    # The genesis workspace root must sit beside, not under, the runtime data root.
    monkeypatch.setenv("OUROBOROS_SUBAGENT_PROJECTS_ROOT", str(tmp_path_factory.mktemp("projects_root")))
    create_project(tmp_path, "racer", name="Racer")
    ref = build_owner_message_ref(chat_id=1, client_message_id="cm-1", ts="2026-09-24T00:00:00+00:00",
                                  text="Repair the skill")
    pending = []
    handler_ctx = SimpleNamespace(
        DRIVE_ROOT=tmp_path, WORKERS={0: SimpleNamespace()}, PENDING=pending, bridge=None,
        append_jsonl=lambda *_a, **_k: None, persist_queue_snapshot=lambda **_k: True,
        enqueue_task=lambda task: pending.append(dict(task)) or pending[-1],
        load_state=lambda: {"owner_chat_id": 1},
    )
    event = {"type": "promote_chat_to_task", "task_id": f"root-{stamp}", "routing_token": "tok-1",
             "objective": "Repair the skill", "chat_id": 1, "client_message_id": "cm-1",
             "project_id": "racer", "routed_from_main": True, "host_initiated": True}
    if stamp == "logged-ref":
        event.update(source_ref=dict(ref), source_text="Repair the skill")
    elif stamp == "suppressed-log":
        event["origin_suppressed"] = True

    outcome = _handle_promote_chat_to_task(event, handler_ctx)

    assert outcome["status"] == "scheduled", outcome
    [task] = pending
    origin = run_origin(task)
    assert origin["owner_ingress"] is (stamp != "none")
    assert origin["source"] == "promote_chat_to_task" and origin["delegation_role"] == "root"
    assert (task.get("origin_message_ref"), (task.get("metadata") or {}).get("origin_suppressed")) == {
        "logged-ref": (ref, None), "suppressed-log": (None, True), "none": (None, None),
    }[stamp]

    # The running-record writer persists the producer's `source`, so a reader of the persisted
    # record under the live metadata (the owner card's host facts, ``core_artifacts._quiz_host_facts``)
    # and the synthesis (the live task) project one and the same origin. A writer that drops
    # `source` fails here.
    from ouroboros.agent import OuroborosAgent
    from ouroboros.task_results import load_task_result

    OuroborosAgent._persist_running_record(SimpleNamespace(env=SimpleNamespace(drive_root=tmp_path)), task)
    live_metadata = {**task["metadata"], **({"origin_message_ref": ref} if stamp == "logged-ref" else {})}
    stored = load_task_result(tmp_path, task["id"])
    persisted = run_origin({**stored, "metadata": {**(stored.get("metadata") or {}), **live_metadata}})
    live = run_origin({**task, "metadata": live_metadata})
    # The record carries no task type (its readers take the live one); every other fact is the same.
    assert persisted == {key: value for key, value in live.items() if key != "task_type"}
    assert persisted["source"] == "promote_chat_to_task"


@pytest.mark.parametrize("label", ["initial_user", "initial_text"])
def test_the_safety_check_sees_the_corpus_label_it_is_asked_to_weigh(label):
    """Safety prints the retained corpus with its labels: the supervisor can tell a
    row the owner door stamped from a run's unattributed initial text without the
    host deciding consent for it."""
    from ouroboros.safety import _build_check_prompt

    ctx = SimpleNamespace(task_id="t-1", project_id="", task_contract={}, task_constraint=None,
                          _owner_directives=[{"source": label, "content": "run the migration"}])
    prompt = _build_check_prompt("shell", {"cmd": "ls"}, ctx=ctx)
    assert f'"source": "{label}"' in prompt and "run the migration" in prompt
    other = "initial_text" if label == "initial_user" else "initial_user"
    assert f'"source": "{other}"' not in prompt
