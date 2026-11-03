"""Unit tests for the platform-free contract layer.

Ported from the Rust unit tests of element.rs/uia.rs: these run on any OS,
without Windows, without comtypes - the whole point of keeping the contract
layer separate.
"""

from __future__ import annotations

import tests._windows_computer_use  # noqa: F401  (registers the payload package)
from windows_computer_use import core, tree  # noqa: E402
from windows_computer_use.core import (  # noqa: E402
    NodeView,
    Selector,
    act_from,
    act_mutating,
    act_plan,
    ambiguous_error,
    choose,
    clip_utf16,
    find_plan,
    find_receipt,
    not_found,
    parse_hwnd,
    read_plan,
    receipt,
    role_name,
    same_fold,
    win32_role,
)


def node(role, name=None, id=None, value=None):
    return NodeView(role=role, name=name, automation_id=id, value=value)


# ── selector matching (element.rs: every_selector_field_must_match_at_once) ────

def test_every_selector_field_must_match_at_once():
    want = Selector(role="button", name_contains="сохран")
    assert want.matches(node("button", "Сохранить как"))
    assert not want.matches(node("menu_item", "Сохранить как"))
    assert not want.matches(node("button", "Отмена"))


def test_a_name_written_off_the_screen_matches_regardless_of_case():
    # Cyrillic is the common case here, not a corner: labels are Russian.
    assert Selector(name="ок").matches(node("button", "ОК"))
    assert Selector(name="СОХРАНИТЬ").matches(node("button", "Сохранить"))
    assert Selector(name="cancel").matches(node("button", "Cancel"))
    assert Selector(name_contains="ХРАНИ").matches(node("button", "Сохранить как"))


def test_an_automation_id_is_compared_exactly():
    assert Selector(automation_id="saveBtn").matches(node("button", id="saveBtn"))
    assert not Selector(automation_id="saveBtn").matches(node("button", id="savebtn"))


def test_exact_fields_keep_whitespace_and_unknown_selectors_fail_closed():
    p = act_plan({"do": "invoke", "select": {"automation_id": " save ", "name": " OK "}})
    assert p.select.automation_id == " save "
    assert p.select.name == " OK "
    try:
        act_plan({"do": "invoke", "select": {"role": "button", "automationId": "save"}})
        raise AssertionError("unknown select field must fail")
    except ValueError as error:
        assert "automationId" in str(error)
    try:
        act_plan({"do": "invoke", "select": {"role": "button"}, "expected_pid": 123})
        raise AssertionError("unknown top-level field must fail")
    except ValueError as error:
        assert "expected_pid" in str(error)


def test_blank_conditions_never_widen_action_or_find():
    for key in ("automation_id", "name", "name_contains", "value_contains"):
        select = {"role": "button", key: "   "}
        for make in (lambda s: act_plan({"do": "invoke", "select": s}), lambda s: find_plan({"select": s})):
            try:
                make(select)
                raise AssertionError(f"blank {key} must fail")
            except ValueError as error:
                assert "blank" in str(error)
    exact = act_plan({"do": "invoke", "select": {"name": " OK "}})
    assert exact.select.name == " OK "


# ── choose (element.rs: choosing_refuses_to_pick_for_her) ──────────────────────

def test_choosing_refuses_to_pick_for_you():
    assert choose([], None).which == "none"
    one = choose([7], None)
    assert one.which == "one" and one.index == 7
    amb = choose([7, 9], None)
    assert amb.which == "ambiguous" and amb.total == 2
    nth = choose([7, 9], 1)
    assert nth.which == "one" and nth.index == 9
    # Asked for the third of two: a miss, not "take the last one".
    assert choose([7, 9], 2).which == "none"


# ── act plan parsing (element.rs tests) ────────────────────────────────────────

def test_an_empty_selector_is_refused_by_name():
    try:
        act_plan({"do": "invoke"})
        raise AssertionError("must refuse")
    except ValueError as error:
        assert "select is empty" in str(error)
        assert "automation_id" in str(error)


def test_an_unknown_action_lists_the_known_ones():
    try:
        act_from("press")
        raise AssertionError("must refuse")
    except ValueError as error:
        assert "invoke" in str(error)
        assert "scroll_into_view" in str(error)


def test_setting_a_value_without_text_is_a_forgotten_argument():
    # A null is refused like a missing key: handed to the pattern it would clear the field.
    for args in ({}, {"text": None}, {"text": 7}):
        try:
            act_plan({"do": "set_value", "select": {"role": "edit"}, **args})
            raise AssertionError("must refuse")
        except ValueError as error:
            assert "empty string to clear" in str(error)
    ok = act_plan({"do": "set_value", "select": {"role": "edit"}, "text": ""})
    assert ok.text == ""


def test_text_offered_to_an_action_that_ignores_it_is_refused():
    try:
        act_plan({"do": "invoke", "select": {"name": "OK"}, "text": "привет"})
        raise AssertionError("must refuse")
    except ValueError as error:
        assert "set_value only" in str(error)


def test_limits_are_clamped_and_never_zero():
    big = act_plan({
        "do": "focus", "select": {"role": "edit"},
        "max_nodes": 999_999, "max_depth": 999, "timeout_ms": 999_999,
    })
    assert big.max_nodes == core.ACT_MAX_NODES_CAP
    assert big.max_depth == core.ACT_MAX_DEPTH_CAP
    assert big.timeout_ms == core.ACT_TIMEOUT_CAP_MS
    zero = act_plan({"do": "focus", "select": {"role": "edit"}, "max_nodes": 0})
    assert zero.max_nodes == 1  # zero nodes is not a walk, it is silence


def test_moving_the_view_is_not_called_a_change_of_data():
    assert not act_mutating("focus")
    assert not act_mutating("scroll_into_view")
    assert act_mutating("invoke")
    assert act_mutating("set_value")


# ── find plan (element.rs find tests) ─────────────────────────────────────────

def test_find_refuses_an_empty_selector_and_points_at_reading_the_window():
    try:
        find_plan({})
        raise AssertionError("must refuse")
    except ValueError as error:
        assert "select is empty" in str(error)
        assert "window_read" in str(error)


def test_find_says_that_nth_is_not_its_word():
    try:
        find_plan({"select": {"role": "button", "nth": 2}})
        raise AssertionError("must refuse")
    except ValueError as error:
        assert "nth belongs to" in str(error)


def test_find_keeps_exact_selector_whitespace_and_refuses_unknown_fields():
    plan = find_plan({"select": {"automation_id": " save ", "name": " OK "}})
    assert plan.select.automation_id == " save "
    assert plan.select.name == " OK "
    try:
        find_plan({"select": {"role": "button"}, "timeout_m": 1000})
        raise AssertionError("must refuse")
    except ValueError as error:
        assert "unknown field" in str(error).lower()


def test_find_waits_only_when_asked():
    quick = find_plan({"select": {"role": "button"}})
    assert quick.timeout_ms == 0
    assert quick.limit == core.FIND_LIMIT_DEFAULT
    waiting = find_plan({"select": {"role": "button"}, "timeout_ms": 999_999})
    assert waiting.timeout_ms == core.ACT_TIMEOUT_CAP_MS
    many = find_plan({"select": {"role": "button"}, "limit": 9_999})
    assert many.limit == core.FIND_LIMIT_CAP


def test_find_receipt_says_when_it_showed_fewer_than_it_found():
    plan = find_plan({"select": {"role": "button"}, "limit": 2})
    said = find_receipt(plan, 7, [{"role": "button"}, {"role": "button"}], 120, True, 0, 1)
    assert said["matched"] == 7
    assert said["shown"] == 2
    assert said["truncated"] is True
    assert said["searched_whole_window"] is True


# ── not_found / ambiguity (element.rs: an_unfinished_walk_is_never_called_absence) ──

def test_an_unfinished_walk_is_never_called_absence():
    reason, hint = not_found("timeout", False, False)
    assert reason == "timeout"
    assert "raise timeout_ms" in hint
    reason, hint = not_found("max_nodes", False, False)
    assert reason == "max_nodes"
    assert "raise max_nodes" in hint
    assert not_found("max_depth", False, False)[0] == "max_depth"
    assert not_found("provider_refused", False, False)[0] == "incomplete"
    reason, hint = not_found(None, False, False)
    assert reason == "not_found"
    assert "window_read" in hint
    assert not_found(None, True, True)[0] == "nth_out_of_range"
    assert not_found(None, True, False)[0] == "not_found"


def test_ambiguity_is_answered_with_the_candidates_themselves():
    said = ambiguous_error(3, [{"name": "OK", "automation_id": "ok1"}])
    assert said["reason"] == "ambiguous"
    assert any("ok1" in str(item) for item in said["candidates"])
    assert "nth" in said["hint"]


def test_the_receipt_does_not_claim_success():
    p = act_plan({"do": "invoke", "select": {"name": "OK"}})
    said = receipt(p, {"role": "button"}, {"role": "button"}, 12, 2)
    assert "element_after" in said
    assert "yours to judge" in said["note"]


# ── read plan / limits (uia.rs tests) ─────────────────────────────────────────

def test_a_cap_out_of_range_is_pulled_to_the_border_and_said_out_loud():
    stretched = read_plan({"max_nodes": 999_999, "timeout_ms": 0})
    assert stretched.limits.max_nodes == core.READ_CEILINGS["max_nodes"]
    assert stretched.limits.timeout_ms == core.READ_FLOORS["timeout_ms"]
    assert any("max_nodes=999999" in note and "5000" in note for note in stretched.notes)
    assert any("timeout_ms=0" in note and "100" in note for note in stretched.notes)
    quiet = read_plan({"max_nodes": 42})
    assert quiet.limits.max_nodes == 42
    assert quiet.notes == []


def test_a_misspelled_argument_is_named_instead_of_ignored():
    try:
        read_plan({"max_node": 10})
        raise AssertionError("must refuse")
    except ValueError as error:
        assert "max_node" in str(error)


def test_shape_and_hwnd_are_parsed_the_same_way_window_list_prints_them():
    assert read_plan({"shape": "flat"}).shape == "flat"
    assert read_plan({}).shape == "tree"
    try:
        read_plan({"shape": "list"})
        raise AssertionError("must refuse")
    except ValueError as error:
        assert "shape must be tree or flat" in str(error)
    assert read_plan({"hwnd": "0x1F4"}).hwnd == 500
    assert read_plan({"hwnd": "500"}).hwnd == 500
    assert read_plan({"hwnd": 500}).hwnd == 500
    try:
        read_plan({"hwnd": 0})
        raise AssertionError("must refuse")
    except ValueError as error:
        assert "invalid hwnd" in str(error)


def test_the_plain_win32_path_can_be_asked_for_by_name():
    assert not read_plan({}).force_win32
    assert not read_plan({"backend": "auto"}).force_win32
    assert read_plan({"backend": "win32"}).force_win32
    try:
        read_plan({"backend": "uia"})
        raise AssertionError("must refuse")
    except ValueError as error:
        assert "backend must be auto or win32" in str(error)


def test_every_named_cap_ships_its_ceiling_and_nothing_else_hides():
    limits = core.ReadLimits().json()
    for name in core.READ_DEFAULTS:
        assert limits[name] <= limits[f"{name}_ceiling"]
    assert limits["worker_grace_ms"] == core.WORKER_GRACE_MS
    assert limits["win32_text_timeout_ms"] == core.WIN32_TEXT_TIMEOUT_MS


# ── clip_utf16 / role_name / win32_role (uia.rs tests) ─────────────────────────

def test_text_is_clipped_by_utf16_units_not_by_characters():
    # A surrogate pair takes two UTF-16 units: at limit 3 one pair + one char fit.
    text, cut = clip_utf16("𐐀𐐀", 3)
    assert text == "𐐀" and cut
    text, cut = clip_utf16("𐐀a", 3)
    assert text == "𐐀a" and not cut
    text, cut = clip_utf16("abc", 3)
    assert text == "abc" and not cut


def test_every_uia_control_type_has_a_word_and_a_stranger_keeps_its_number():
    for control_id in range(50000, 50041):
        assert not role_name(control_id).startswith("control_type_"), control_id
    assert role_name(0) == "unknown"
    assert role_name(50041) == "control_type_50041"
    assert role_name(50000) == "button"
    assert role_name(50002) == "check_box"


def test_win32_classes_map_to_readable_roles():
    assert win32_role("Button") == "button"
    assert win32_role("RichEdit50W") == "edit"
    assert win32_role("Edit") == "edit"
    assert win32_role("Static") == "text"
    assert win32_role("#32770") == "dialog"
    assert win32_role("Chrome_WidgetWin_1") == "window"


# ── render (uia.rs tree tests, ported against tree.render) ─────────────────────

def _sample_walk():
    def n(parent, depth, role, name):
        return tree.Node(parent=parent, depth=depth, role=role, name=name or None)

    walk = tree.Walk()
    walk.nodes = [
        n(None, 0, "window", "Notepad"),
        n(0, 1, "menu_bar", "Menu"),
        n(1, 2, "menu_item", "Save"),
        n(0, 1, "edit", "Document"),
    ]
    walk.depth_reached = 2
    return walk


def _window():
    return tree.WindowInfo(
        hwnd=0x1234, title="Notepad", cls="Notepad", pid=42,
        rect=tree.Rect(0, 0, 100, 50), foreground=True, resolved_from="foreground", dpi=96,
    )


def _rendered(plan, walk, **kw):
    return tree.render(plan, _window(), walk, backend="uia",
                       backend_detail="detail", fallback_reason=None,
                       elapsed_ms=17, **kw)


def test_a_tree_answer_keeps_parent_links_and_child_order():
    value = _rendered(read_plan({}), _sample_walk())
    assert value["shape"] == "tree"
    assert value["nodes_returned"] == 4
    assert value["window"]["hwnd"] == "0x1234"
    root = value["root"]
    assert root["role"] == "window"
    assert root["children"][0]["name"] == "Menu"
    assert root["children"][0]["children"][0]["name"] == "Save"
    assert root["children"][1]["name"] == "Document"
    assert root["children"][1]["parent"] == 0


def test_coordinates_come_with_a_point_that_can_actually_be_clicked():
    walk = _sample_walk()
    walk.nodes[3].rect = tree.Rect(10, 20, 110, 60)
    value = _rendered(read_plan({"shape": "flat"}), walk)
    edit = value["items"][3]
    assert edit["rect"]["width"] == 100
    assert edit["rect"]["height"] == 40
    assert edit["rect"]["center"]["x"] == 60
    assert edit["rect"]["center"]["y"] == 40


def test_a_filter_keeps_the_path_in_a_tree_and_only_the_hits_when_flat():
    walk = _sample_walk()
    tree_value = _rendered(read_plan({"text_contains": "save"}), walk)
    assert tree_value["nodes_returned"] == 3
    assert tree_value["root"]["children"][0]["children"][0]["name"] == "Save"
    assert len(tree_value["root"]["children"]) == 1
    assert tree_value["filter"]["text_contains"] == "save"

    flat = _rendered(read_plan({"shape": "flat", "text_contains": "save"}), walk)
    assert flat["nodes_returned"] == 1
    assert len(flat["items"]) == 1
    assert flat["items"][0]["name"] == "Save"

    empty = _rendered(read_plan({"text_contains": "нетакого"}), walk)
    assert empty["nodes_returned"] == 0
    assert empty["root"] is None
    assert empty["nodes_read"] == 4


def test_a_whole_reading_says_so_and_a_cut_one_says_how_much_is_left():
    plan = read_plan({})
    whole = _rendered(plan, _sample_walk())
    assert whole["total_known"] is True
    assert whole["truncated"] is False
    assert whole["truncated_by"] == []
    assert whole["notes"] == []

    cut = _sample_walk()
    cut.mark("max_nodes")
    cut.discovered_unread = 212
    value = _rendered(plan, cut)
    assert value["total_known"] is False
    assert value["truncated"] is True
    assert value["truncated_by"][0] == "max_nodes"
    assert value["discovered_unread"] == 212
    assert any(
        "max_nodes=400" in note and "212 more" in note for note in value["notes"]
    )


def test_an_unknown_state_is_absent_and_never_printed_as_false():
    walk = _sample_walk()
    walk.nodes[2].enabled = False
    walk.nodes[2].checked = "on"
    value = _rendered(read_plan({"shape": "flat"}), walk)
    save = value["items"][2]
    assert save["state"]["enabled"] is False
    assert save["state"]["checked"] == "on"
    assert "selected" not in save["state"]
    assert "expanded" not in save["state"]
    assert "state" not in value["items"][1]
    assert "does not mean false" in value["state_semantics"]


def test_a_truncated_label_is_marked_on_the_node_that_lost_it():
    walk = _sample_walk()
    walk.nodes[3].text_truncated = True
    walk.nodes[3].children_unread = "max_depth"
    value = _rendered(read_plan({"shape": "flat"}), walk)
    assert value["items"][3]["text_truncated"] is True
    assert value["items"][3]["children_unread"] == "max_depth"
    assert "text_truncated" not in value["items"][0]


def test_a_fallback_answer_is_an_answer_and_names_why_it_is_the_lesser_one():
    plan = read_plan({})
    walk = tree.Walk()
    walk.nodes = [tree.Node(parent=None, depth=0, role="window", name="Быстрые настройки")]
    value = tree.render(
        plan, _window(), walk, backend="win32", backend_detail="d",
        fallback_reason="UI Automation did not answer within timeout_ms=1800",
        elapsed_ms=2,
        extra_notes=["Chromium, Electron, WPF and UWP surfaces look empty here"],
        workers_live=1,
    )
    assert value["ok"] is True
    assert value["backend"] == "win32"
    assert value["nodes_returned"] == 1
    assert value["uia_workers_live"] == 1
    assert "did not answer" in value["fallback_reason"]
    assert any("look empty here" in note for note in value["notes"])


def test_same_fold_handles_none_and_cyrillic():
    assert same_fold("ОК", "ок")
    assert same_fold("Cancel", "cancel")
    assert not same_fold(None, "x")


def test_parse_hwnd_rejects_garbage():
    for bad in (0, "abc", "", -5, True):
        try:
            parse_hwnd(bad)
            raise AssertionError(f"{bad!r} must be refused")
        except ValueError:
            pass


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as error:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {error}")
    raise SystemExit(1 if failures else 0)
