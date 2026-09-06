"""v6.37.0 guard (C3.1/C3.3): the parent's 'you may delegate / mutate / fan out
further' intent must propagate STRUCTURALLY via a typed delegation_budget on the
task contract and be surfaced in the child's prompt — instead of being lost in
freeform objective prose (the cyber-racing 'maximum subagents' request that
collapsed into 3 flat research leaves)."""

from types import SimpleNamespace


def test_delegation_budget_defaults_and_normalization():
    from ouroboros.contracts.task_contract import build_task_contract, normalize_delegation_budget

    c = build_task_contract({"objective": "x"})
    b = c["delegation_budget"]
    assert b["may_delegate"] is True
    assert b["may_mutate"] is False  # mutation is opt-in
    assert b["may_fan_out"] is True
    assert b["depth_remaining"] is None and b["max_children"] is None
    assert b["intent_note"] == ""

    c2 = build_task_contract({
        "objective": "x",
        "delegation_budget": {"may_mutate": "yes", "depth_remaining": "2", "intent_note": "  go deep  "},
    })
    b2 = c2["delegation_budget"]
    assert b2["may_mutate"] is True
    assert b2["depth_remaining"] == 2
    assert b2["intent_note"] == "go deep"

    # junk is coerced safely
    assert normalize_delegation_budget(None)["may_delegate"] is True
    assert normalize_delegation_budget({"depth_remaining": "junk"})["depth_remaining"] is None
    assert normalize_delegation_budget({"depth_remaining": -5})["depth_remaining"] == 0


def test_compose_subagent_text_surfaces_budget():
    from supervisor.events import _compose_subagent_text

    txt = _compose_subagent_text(
        "obj", role="builder", expected_output="out", constraints="", context="",
        delegation_budget={
            "may_delegate": True, "may_mutate": True, "may_fan_out": True,
            "depth_remaining": 2, "max_children": None,
            "intent_note": "build the whole game, delegate per subsystem",
        },
    )
    assert "[DELEGATION BUDGET]" in txt
    assert "depth_remaining=2" in txt
    assert "mutating descendants permitted" in txt
    assert "build the whole game, delegate per subsystem" in txt

    # no budget -> no section (back-compat with callers that don't pass one)
    txt2 = _compose_subagent_text("obj", role="r", expected_output="out", constraints="", context="")
    assert "[DELEGATION BUDGET]" not in txt2


def test_absorption_full_then_whole_pointer_and_grandchild_rollup():
    from ouroboros.task_status import format_subagent_absorption_message
    children = [
        {"task_id": "d1", "parent_task_id": "P", "status": "completed", "role": "a", "result": "A" * 50, "cost_usd": 0.1},
        {"task_id": "d2", "parent_task_id": "P", "status": "completed", "role": "b", "result": "B" * 5000, "cost_usd": 0.2},
        {"task_id": "gc1", "parent_task_id": "d1", "status": "completed", "role": "c", "result": "grandchild-secret"},
        {"task_id": "d3", "parent_task_id": "P", "status": "running", "role": "d", "result": ""},
    ]
    msg = format_subagent_absorption_message(children, parent_task_id="P", budget_chars=100)
    assert "[SUBAGENT_RESULTS" in msg
    assert "A" * 50 in msg                      # d1 injected in FULL (fits)
    assert "B" * 5000 not in msg                # d2 over budget -> NOT injected
    assert 'get_task_result("d2")' in msg       # d2 replaced WHOLE by a pointer
    assert "grandchild-secret" not in msg       # grandchild raw output rolled up, not in root
    assert "DEEPER DESCENDANTS" in msg and "gc1" in msg
    assert "STILL RUNNING" in msg and "d3" in msg

    msg2 = format_subagent_absorption_message(children, parent_task_id="P", budget_chars=1_000_000)
    assert "B" * 5000 in msg2                    # generous budget -> both full


def test_absorption_digest_carries_a_childs_typed_custody_debt():
    """A parent could absorb its child's work while the child's own delegated
    patch sat undisposed, and the digest said nothing. A clean child stays
    noise-free."""
    from ouroboros.task_status import format_subagent_absorption_message
    children = [
        {"task_id": "d1", "parent_task_id": "P", "status": "completed", "role": "a",
         "result": "work", "delegated_runs_unreconciled": ["patch:run-x"]},
        {"task_id": "d2", "parent_task_id": "P", "status": "completed", "role": "b",
         "result": "clean work"},
    ]
    msg = format_subagent_absorption_message(children, parent_task_id="P")
    d1_header = next(l for l in msg.splitlines() if l.startswith("## child d1"))
    d2_header = next(l for l in msg.splitlines() if l.startswith("## child d2"))
    assert "custody_debt=patch:run-x" in d1_header, d1_header
    assert "custody_debt" not in d2_header, d2_header


def test_absorption_digest_bounds_a_long_custody_debt_list():
    from ouroboros.task_status import format_subagent_absorption_message
    debt = [f"patch:run-{n}" for n in range(12)]
    children = [{"task_id": "d1", "parent_task_id": "P", "status": "completed",
                 "role": "a", "result": "work",
                 "delegated_runs_unreconciled": debt}]
    header = next(
        l for l in format_subagent_absorption_message(
            children, parent_task_id="P").splitlines()
        if l.startswith("## child d1"))
    assert "custody_debt=" + ",".join(debt[:10]) in header, header
    assert "patch:run-10" not in header and "patch:run-11" not in header, header
    assert "(+2 more" in header and 'get_task_result("d1")' in header, header


def test_child_budget_never_widens_beyond_restrictive_parent():
    """C3.1 narrowing (triad+scope round-2): a child budget must AND every authority
    with the parent's, so a parent that disabled delegation/mutation/fan-out can never
    hand a child MORE authority than it holds (even if the child request asks for it)."""
    from ouroboros.tools.control_delegation import _narrow_child_delegation_budget

    restrictive_parent = {
        "may_delegate": False, "may_mutate": False, "may_fan_out": False,
        "depth_remaining": 3, "max_children": 2,
    }
    child = _narrow_child_delegation_budget(
        restrictive_parent,
        child_depth_remaining=2,        # depth allows it...
        may_mutate=True, may_fan_out=True, max_children=99,  # ...and the request asks for everything
        intent_note="spawn a huge swarm",
    )
    assert child["may_delegate"] is False   # parent said no -> stays no despite depth>0
    assert child["may_mutate"] is False
    assert child["may_fan_out"] is False
    assert child["max_children"] == 2       # capped to the parent's positive cap, not 99


def test_child_budget_unrestricted_parent_honors_request():
    """A legacy/permissive parent (no budget keys) defaults to unrestricted, so the
    child request is honored (backward-compatible pre-C3.1 behavior)."""
    from ouroboros.tools.control_delegation import _narrow_child_delegation_budget

    child = _narrow_child_delegation_budget(
        {},  # legacy contract, no delegation_budget keys
        child_depth_remaining=2,
        may_mutate=True, may_fan_out=True, max_children=5,
        intent_note="",
    )
    assert child["may_delegate"] is True
    assert child["may_mutate"] is True
    assert child["may_fan_out"] is True
    assert child["max_children"] == 5


def test_child_budget_no_delegate_when_depth_exhausted():
    """Even an unrestricted parent yields may_delegate=False once depth is exhausted."""
    from ouroboros.tools.control_delegation import _narrow_child_delegation_budget

    child = _narrow_child_delegation_budget(
        {"may_delegate": True}, child_depth_remaining=0,
        may_mutate=False, may_fan_out=True, max_children=0, intent_note="",
    )
    assert child["may_delegate"] is False


def test_root_honors_explicit_mutative_grant_despite_default_false():
    """Round-3 fix: a ROOT scheduler (parent_is_subagent=False) honors an explicit
    schedule_subagent(may_mutate=True) even though the default contract budget is
    may_mutate=False ('mutation is opt-in') — the AND-narrowing must NOT strip it."""
    from ouroboros.tools.control_delegation import _narrow_child_delegation_budget

    default_root_budget = {"may_delegate": True, "may_mutate": False, "may_fan_out": True}
    child = _narrow_child_delegation_budget(
        default_root_budget, child_depth_remaining=2,
        may_mutate=True, may_fan_out=True, max_children=0, intent_note="",
        parent_is_subagent=False,
    )
    assert child["may_mutate"] is True   # the root's explicit opt-in is honored


def test_subagent_cannot_escalate_mutation():
    """A read-only SUBAGENT (parent_is_subagent=True, may_mutate=False) cannot escalate
    by spawning a mutative descendant — may_mutate stays AND-ed with the parent's."""
    from ouroboros.tools.control_delegation import _narrow_child_delegation_budget

    readonly_subagent = {"may_delegate": True, "may_mutate": False, "may_fan_out": True}
    child = _narrow_child_delegation_budget(
        readonly_subagent, child_depth_remaining=2,
        may_mutate=True, may_fan_out=True, max_children=0, intent_note="",
        parent_is_subagent=True,
    )
    assert child["may_mutate"] is False


def test_intent_note_preserves_complete_delegation_authority():
    """The normalized contract cannot bind a prefix as the complete nesting intent."""
    from ouroboros.contracts.task_contract import normalize_delegation_budget

    intent = "x" * 800 + "THREE_LEVEL_DECISIVE_TAIL"
    b = normalize_delegation_budget({"intent_note": intent})
    assert b["intent_note"] == intent
    assert "OMISSION NOTE" not in b["intent_note"]


def test_child_budget_strict_boolean_parsing():
    """Round-5 fix: the per-call may_mutate/may_fan_out grants must parse via the
    strict normalize_bool — a tool call passing the STRING "false" must NOT be
    treated as True (bool("false") is truthy and would silently grant authority)."""
    from ouroboros.tools.control_delegation import _narrow_child_delegation_budget

    child = _narrow_child_delegation_budget(
        {}, child_depth_remaining=2,
        may_mutate="false", may_fan_out="false", max_children=0, intent_note="",
        parent_is_subagent=False,
    )
    assert child["may_mutate"] is False
    assert child["may_fan_out"] is False
    # and a genuine truthy string is honored
    child2 = _narrow_child_delegation_budget(
        {}, child_depth_remaining=2,
        may_mutate="true", may_fan_out=True, max_children=0, intent_note="",
        parent_is_subagent=False,
    )
    assert child2["may_mutate"] is True
    assert child2["may_fan_out"] is True


def test_requested_depth_is_recorded_as_telemetry_and_never_narrows_the_cap():
    """A nanny chain had no way to SAY how deep it meant to nest, so the root's
    depth summary read "request unknown" and a shallow outcome could not be told
    apart from a reduced capability. The request is now recorded — and recording
    it must not become a second cap: asking for less than the configured depth
    used to narrow what the descendants were permitted."""
    from ouroboros.depth_evidence import build_depth_summary
    from ouroboros.tools.control_delegation import child_budget_for_schedule

    root = {"delegation_budget": {"may_delegate": True}}

    def _schedule(**kwargs):
        return child_budget_for_schedule(
            root, current_depth=0, new_depth=1, max_depth=4, may_mutate=False,
            may_fan_out=True, max_children=0, intent_note="", **kwargs,
        )["depth_provenance"]

    asked = _schedule(requested_depth=2)
    assert asked["requested_depth"] == 2
    assert asked["permitted_depth"] == 4, "a request is telemetry, not a cap"
    assert asked["attempted_depth"] == 1
    assert asked["achieved_depth"] is None

    # Absent or zero means no request of record — never a recorded zero.
    assert _schedule()["requested_depth"] is None
    assert _schedule(requested_depth=0)["requested_depth"] is None
    # The immutable host ceiling bounds authority, not the attested request.
    over_cap = _schedule(requested_depth=99)
    assert over_cap["requested_depth"] == 99
    assert over_cap["permitted_depth"] == 4
    assert build_depth_summary(
        {"delegation_budget": {"depth_provenance": over_cap}}, [],
    )["status"] == "capability_reduced"
    # A non-integer fails SOFT: the handler validates argument names, not value
    # types, and refusing a whole schedule over a telemetry field would trade a
    # real capability for bookkeeping.
    assert _schedule(requested_depth="x")["requested_depth"] is None


def test_an_inherited_request_of_record_survives_a_nannys_own_number():
    from ouroboros.tools.control_delegation import child_budget_for_schedule

    parent = {"delegation_budget": {
        "may_delegate": True, "depth_provenance": {"requested_depth": 3, "permitted_depth": 4},
    }}
    provenance = child_budget_for_schedule(
        parent, current_depth=1, new_depth=2, max_depth=4, may_mutate=False,
        may_fan_out=True, max_children=0, intent_note="", requested_depth=1,
    )["depth_provenance"]
    assert provenance["requested_depth"] == 3
    assert provenance["permitted_depth"] == 4


def test_assignment_stamp_without_a_permitted_fact_uses_the_configured_cap():
    """The stamp used to fold the REQUEST into the permitted number, so a branch
    that asked for two lost the depth the configuration actually allowed."""
    from ouroboros.tools.control_delegation import stamp_task_assignment_depth

    task = {
        "depth": 1,
        "task_contract": {"delegation_budget": {
            "depth_provenance": {"requested_depth": 2, "permitted_depth": "not-a-number"},
        }},
    }
    stamped = stamp_task_assignment_depth(task, max_depth=4)
    provenance = stamped["task_contract"]["delegation_budget"]["depth_provenance"]
    assert provenance["requested_depth"] == 2
    assert provenance["permitted_depth"] == 4


def test_the_swarm_rollup_reports_requested_permitted_and_achieved_depth(tmp_path):
    from ouroboros.task_finalization import build_swarm_efficiency
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result
    from ouroboros.utils import append_jsonl

    root_id = "depth-root"
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    append_jsonl(tmp_path / "logs" / "events.jsonl", {
        "type": "swarm_fanout", "parent_task_id": root_id, "task_ids": ["c1"],
        "requested_count": 1, "ts": "2026-09-03T10:00:00Z",
    })
    write_task_result(
        tmp_path, "c1", STATUS_COMPLETED, result="done",
        parent_task_id=root_id, root_task_id=root_id, delegation_role="subagent",
        depth_provenance={
            "requested_depth": 2, "permitted_depth": 4,
            "attempted_depth": 1, "achieved_depth": 1,
        },
    )
    rollup = build_swarm_efficiency(
        SimpleNamespace(drive_root=tmp_path),
        {
            "id": root_id, "budget_drive_root": str(tmp_path),
            "task_contract": {"delegation_budget": {
                "depth_provenance": {"requested_depth": 2, "permitted_depth": 4},
            }},
        },
    )
    assert rollup["subagent_count"] == 1
    assert rollup["depth"]["requested_depth"] == 2
    assert rollup["depth"]["permitted_depth"] == 4
    assert rollup["depth"]["achieved_depth"] == 1
    assert rollup["depth"]["status"] == "chosen_shallower"
    assert rollup["depth"]["host_visible_only"] is True


def test_a_plain_task_gets_no_depth_block(tmp_path):
    from ouroboros.task_finalization import build_swarm_efficiency

    assert build_swarm_efficiency(
        SimpleNamespace(drive_root=tmp_path), {"id": "lonely"},
    ) is None
