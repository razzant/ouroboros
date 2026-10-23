"""Authorship tripwire for runtime chat sends, not a semantic text classifier.

Every send call and literal chat/send_message envelope in the runtime roots is
examined. Host producers stamp both fields; dynamic transports and model/owner
speech need an exact, counted exception. New calls inside an excepted function
still fail. Arbitrary aliasing/computed envelopes remain a code-review duty.

A second tripwire counts the stamped types (DESIGN "Chat authorship and System
rows"): every host type sits in one closed class with its exact number of sites,
so a new type or a new site of a known one waits for a person to place it.
"""
import ast
import functools
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
# (path, qualified function, shape): (number of sites, reason)
EXCEPTIONS = {
    ("ouroboros/live_conversation.py", "ConversationHost.caption.write", "envelope"): (1, "Projects the durable owner/model/participant caption row with its recorded role and system_type; test_live_host_captions pins both spoken voices."),
    ("supervisor/message_ingress.py", "handle_web_message", "envelope"): (1, "Echoes the accepted canonical row: owner words remain user, host call openings retain system/call_open; test_call_open_ingress pins both paths."),
    ("ouroboros/upgrade_notices.py", "startup_upgrade_notices", "call"): (1, "Host notice uses literal system role and a type from its finite owed-notice tuples; test_reviewer_delivery_defaults and test_legacy_memory_notice pin all three emitted types."),
    ("supervisor/message_bus.py", "_send_markdown", "call"): (1, "Forwards the caller's role/type to the bridge."),
    ("supervisor/message_bus.py", "send_with_budget", "call"): (1, "Forwards caller authorship, not a producer of new text."),
    ("supervisor/message_bus.py", "send_with_budget", "envelope"): (1, "Persists the caller's role/type for progress history."),
    ("supervisor/message_bus.py", "LocalChatBridge.send_message", "envelope"): (1, "Wire projection preserves caller role/type; legacy assistant default is intentional."),
    ("supervisor/message_bus.py", "LocalChatBridge.send_quiz", "envelope"): (1, "Relays the model-authored question pointer with its recorded voice."),
    ("supervisor/events_chat_delivery.py", "_send_message", "call"): (1, "Relays producer role/type without classifying content."),
    ("supervisor/events_chat_delivery.py", "replay_deferred_deliveries", "envelope"): (1, "Continues the stored envelope with its original role/type, without producing new text."),
    ("supervisor/terminal_delivery.py", "replay_pending_deliveries", "envelope"): (1, "Replays the exact stored envelope, including legacy terminal notices (#1010)."),
    ("supervisor/terminal_delivery.py", "build_completed_result_event", "envelope"): (1, "Model final or legacy result; project_terminal_result_event owns terminal provenance (#1010 unchanged)."),
    ("ouroboros/agent_task_pipeline.py", "emit_task_results", "envelope"): (1, "Final model output; prepare_terminal_send_event applies host terminal provenance separately."),
    ("ouroboros/agent.py", "OuroborosAgent._emit_progress", "envelope"): (1, "Explicit narration fact selects model versus host; end-to-end voice test pins both branches."),
    ("ouroboros/tools/control_runtime.py", "_send_user_message", "envelope"): (1, "The model authors this nonterminal reply; proactive_message is not a System voice."),
    ("ouroboros/presence_owner_contact.py", "send_main_message", "envelope"): (1, "The model authors the private Main report; the host source signature attributes it without changing its voice."),
    ("ouroboros/owner_relay.py", "publish_relay", "call"): (1, "One relay row, drawn as a note card: the teller's own note and summary under the owner's host-held quoted words; no host-authored text."),
}


def _literal_values(node, constants):
    if isinstance(node, ast.Constant):
        return {node.value} if isinstance(node.value, str) else set()
    if isinstance(node, ast.Name):
        return constants.get(node.id, set())
    if isinstance(node, ast.IfExp):
        return _literal_values(node.body, constants) | _literal_values(node.orelse, constants)
    return set()


def voice_sites(source):
    """Yield (function, shape, line, fields); fail on invalid syntax, never skip."""
    tree = ast.parse(source)
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = _literal_values(node.value, {})
    found = []

    class Visitor(ast.NodeVisitor):
        scope = ()

        def visit_scope(self, node):
            previous = self.scope
            self.scope = (*previous, node.name)
            self.generic_visit(node)
            self.scope = previous

        visit_FunctionDef = visit_scope
        visit_AsyncFunctionDef = visit_scope
        visit_ClassDef = visit_scope

        def visit_Call(self, node):
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            if name in {"send_message", "send_with_budget"}:
                fields = {kw.arg: kw.value for kw in node.keywords if kw.arg}
                found.append((".".join(self.scope), "call", node.lineno, fields))
            self.generic_visit(node)

        def visit_Dict(self, node):
            fields = {key.value: value for key, value in zip(node.keys, node.values)
                      if isinstance(key, ast.Constant) and isinstance(key.value, str)}
            if _literal_values(fields.get("type"), constants) & {"chat", "send_message"}:
                found.append((".".join(self.scope), "envelope", node.lineno, fields))
            self.generic_visit(node)

    Visitor().visit(tree)
    return found, constants


def unstamped_sites(sources, exceptions):
    violations, observed = [], Counter()
    for path, source in sources.items():
        sites, constants = voice_sites(source)
        for function, shape, line, fields in sites:
            key = (path, function, shape)
            # Count EVERY occurrence, including stamped ones: adding a sibling
            # inside a transport cannot inherit an existing exception.
            if key in exceptions:
                observed[key] += 1
                continue
            role = _literal_values(fields.get("role"), constants)
            kind = _literal_values(fields.get("system_type"), constants)
            if role == {"user"}:
                continue  # owner ingress/echo, not agent or host speech
            if role != {"system"} or not kind or "" in kind:
                violations.append(f"{path}:{line} {function} ({shape})")
    for key, (count, reason) in exceptions.items():
        if observed[key] != count or not reason.strip():
            violations.append(f"exception {key}: expected {count}, observed {observed[key]}")
    return violations


def _runtime_sources():
    paths = [REPO / "server.py", *(REPO / "ouroboros").rglob("*.py"),
             *(REPO / "supervisor").rglob("*.py")]
    return {p.relative_to(REPO).as_posix(): p.read_text(encoding="utf-8") for p in paths}


def test_all_runtime_chat_producers_declare_authorship():
    sources = _runtime_sources()
    assert not (missing := unstamped_sites(sources, EXCEPTIONS)), (
        f"Chat sites need role='system' AND a nonempty system_type, or an exact "
        f"reasoned model/transport exception: {missing}")


@pytest.mark.parametrize("call", [
    'send_with_budget(1, "host")',
    'ctx.send_with_budget(1, "host", role="system")',
    'bridge.send_message(1, "host", system_type="notice")',
    'ctx.send_with_budget(1, "host", role="system", system_type="")',
    'q.put({"type":"send_message", "text":"host"})',
    'frame = {"type":"chat", "role":"assistant", "content":"host"}',
])
def test_a_new_unstamped_site_is_red(call):
    assert unstamped_sites({"new.py": "def emit():\n    " + call}, {})


def test_stamps_constants_and_owner_echo_are_positive_paths():
    source = '''KIND = "notice"
def emit():
    ctx.send_with_budget(1, "host", role="system", system_type=KIND)
    q.put({"type":"send_message", "role":"system", "system_type":"notice"})
    frame = {"type":"photo" if image else "chat", "role":"user"}
'''
    assert not unstamped_sites({"new.py": source}, {})


def test_exceptions_are_counted_and_stale_rows_fail():
    exceptions = {("new.py", "relay", "call"): (1, "Forwards model text")}
    source = 'def relay():\n    bridge.send_message(1, text)\n'
    assert not unstamped_sites({"new.py": source}, exceptions)
    assert unstamped_sites({"new.py": source + '    bridge.send_message(1, "host")\n'}, exceptions)
    assert unstamped_sites({"new.py": ""}, exceptions)


# A host fact names its carrier: a task's fact is a row of its card, an owner message's fact the
# typed annotation on it, and a landed act publishes no row. A standalone System row exists only
# where nothing else can carry the fact. Each literal host type: (class, number of producer sites).
# Two types are mixed and disclosed: `deep_self_review_queued` also echoes the model's own request,
# `skill_repair` also the host's automatic repair after a marketplace install. Nothing produces an
# `acceptance_late_settlement` row since the review fold; the old rows still render as card rows.
HOST_ROW_CLASSES = {
    "budget_notice": ("runtime_state", 1),
    "cancel_receipt": ("owner_act_receipt", 2),
    "command_reply": ("owner_act_receipt", 2),
    "consciousness_notice": ("runtime_state", 1),
    "deep_self_review_queued": ("owner_act_receipt", 1),
    "deep_self_review_unavailable": ("not_started", 2),
    "evolution_cycle_outcome": ("runtime_state", 1),
    "evolution_notice": ("runtime_state", 10),
    "finalization_notice": ("card_row", 2),
    "host_progress": ("card_row", 1),
    "initialization_notice": ("not_started", 1),
    "managed_update_notice": ("runtime_state", 2),
    "model_capability_notice": ("runtime_state", 1),
    "owner_pause_notice": ("card_row", 1),
    "project_completion_summary": ("main_lifecycle", 1),
    "project_handoff": ("main_lifecycle", 1),
    "project_started": ("main_lifecycle", 1),
    "promotion_notice": ("runtime_state", 4),
    "reminder": ("signed_words", 1),
    "restart_notice": ("runtime_state", 7),
    "rollback_notice": ("runtime_state", 1),
    "retired_settings_notice": ("runtime_state", 2),
    "review_pool_migration_notice": ("runtime_state", 1),
    "runtime_liveness_notice": ("runtime_state", 1),
    "skill_lifecycle": ("card_row", 2),
    "skill_lifecycle_pointer": ("owner_act_receipt", 1),
    "skill_notice": ("signed_words", 1),
    "skill_repair": ("owner_act_receipt", 1),  # the owner's Repair; marketplace auto-repair is retired
    "startup_notice": ("runtime_state", 2),
    "steer_not_delivered": ("not_started", 1),
    "storage_switch_notice": ("runtime_state", 1),
    "subagent_admission_notice": ("card_row", 1),
    "subagent_started": ("card_row", 1),
    "subagent_terminal_notice": ("card_row", 1),
    "supervisor_failure": ("no_mind", 1),
    "task_admission_notice": ("not_started", 3),
    "task_error": ("no_mind", 1),
    "task_not_started": ("not_started", 3),
    "task_pause_notice": ("card_row", 1),
    "task_reaper_notice": ("card_row", 4),
    "task_scheduled": ("card_row", 1),
    "task_start_unconfirmed": ("not_started", 1),
    "terminal_incident": ("no_mind", 2),
    "ui_hold": ("runtime_state", 1),
    "worker_failure": ("no_mind", 2),
    "worker_readiness_notice": ("runtime_state", 2),
}
# The closed classes; none is "the success of a landed act" (its receipt or card already says it).
HOST_ROW_CLASS_NAMES = {"no_mind", "not_started", "owner_act_receipt", "runtime_state", "main_lifecycle",
                        "card_row", "signed_words"}


def host_row_census(sources):
    """The stamped literal host types and their producer sites (transports and owner echo excluded)."""
    observed = Counter()
    for path, source in sources.items():
        sites, constants = voice_sites(source)
        for function, shape, _line, fields in sites:
            if (path, function, shape) in EXCEPTIONS or _literal_values(fields.get("role"), constants) == {"user"}:
                continue
            observed.update(_literal_values(fields.get("system_type"), constants) - {""})
    return observed


def census_errors(observed, classes=HOST_ROW_CLASSES):
    errors = [f"{kind}: {count} site(s), none classified" for kind, count in sorted(observed.items())
              if kind not in classes]
    errors += [f"{kind}: classified with {want} site(s), {observed.get(kind, 0)} found"
               for kind, (_name, want) in sorted(classes.items()) if observed.get(kind, 0) != want]
    return errors + [f"{kind}: unknown class {name!r}" for kind, (name, _want) in sorted(classes.items())
                     if name not in HOST_ROW_CLASS_NAMES]


@functools.lru_cache(maxsize=1)
def _live_census():
    return host_row_census(_runtime_sources())


def test_every_host_row_type_sits_in_one_closed_class_with_its_counted_sites():
    assert not (errors := census_errors(_live_census())), (
        "Place each host row in DESIGN 'Chat authorship and System rows' (a landed act publishes no row; "
        f"a task's fact is a row of its card) and record its class and site count here: {errors}")


@pytest.mark.parametrize("call", [
    'ctx.send_with_budget(1, "success report", role="system", system_type="task_scheduled")',
    'ctx.send_with_budget(1, "Repair task accepted", role="system", system_type="repair_accepted")',
    'q.put({"type": "send_message", "text": "started", "role": "system", "system_type": "reminder"})',
])
def test_a_new_site_of_a_known_type_or_a_new_type_is_red(call):
    """Borrowing an approved type (a card type sent unplaced) is a new site, so it is red too."""
    assert census_errors(_live_census() + host_row_census({"new.py": "def emit():\n    " + call}))
    assert census_errors(_live_census()) == []
