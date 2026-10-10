"""Authorship tripwire for runtime chat sends, not a semantic text classifier.

Every send call and literal chat/send_message envelope in the runtime roots is
examined. Host producers stamp both fields; dynamic transports and model/owner
speech need an exact, counted exception. New calls inside an excepted function
still fail. Arbitrary aliasing/computed envelopes remain a code-review duty.
"""
import ast
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
# (path, qualified function, shape): (number of sites, reason)
EXCEPTIONS = {
    ("ouroboros/upgrade_notices.py", "startup_upgrade_notices", "call"): (1, "Host notice uses literal system role and a type from its finite owed-notice tuples; test_reviewer_delivery_defaults and test_legacy_memory_notice pin all three emitted types."),
    ("supervisor/message_bus.py", "_send_markdown", "call"): (1, "Forwards the caller's role/type to the bridge."),
    ("supervisor/message_bus.py", "send_with_budget", "call"): (1, "Forwards caller authorship, not a producer of new text."),
    ("supervisor/message_bus.py", "send_with_budget", "envelope"): (1, "Persists the caller's role/type for progress history."),
    ("supervisor/message_bus.py", "LocalChatBridge.send_message", "envelope"): (1, "Wire projection preserves caller role/type; legacy assistant default is intentional."),
    ("supervisor/message_bus.py", "LocalChatBridge.send_quiz", "envelope"): (1, "Relays the model-authored question pointer with its recorded voice."),
    ("supervisor/events_chat_delivery.py", "_handle_send_message", "call"): (1, "Relays producer role/type without classifying content."),
    ("supervisor/terminal_delivery.py", "replay_pending_deliveries", "envelope"): (1, "Replays the exact stored envelope, including legacy terminal notices (#1010)."),
    ("supervisor/terminal_delivery.py", "build_completed_result_event", "envelope"): (1, "Model final or legacy result; project_terminal_result_event owns terminal provenance (#1010 unchanged)."),
    ("ouroboros/agent_task_pipeline.py", "emit_task_results", "envelope"): (1, "Final model output; prepare_terminal_send_event applies host terminal provenance separately."),
    ("ouroboros/agent.py", "OuroborosAgent._emit_progress", "envelope"): (1, "Explicit narration fact selects model versus host; end-to-end voice test pins both branches."),
    ("ouroboros/tools/control_runtime.py", "_send_user_message", "envelope"): (1, "The model authors this nonterminal reply; proactive_message is not a System voice."),
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


def test_all_runtime_chat_producers_declare_authorship():
    paths = [REPO / "server.py", *(REPO / "ouroboros").rglob("*.py"),
             *(REPO / "supervisor").rglob("*.py")]
    sources = {p.relative_to(REPO).as_posix(): p.read_text(encoding="utf-8") for p in paths}
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
