"""Overlay two static responses in variant B only, never patch the checkout."""
from tests.test_subscription_setup_browser import WEB

from .scene import HERE


def once(source, needle, replacement):
    if source.count(needle) != 1:
        raise ValueError(f"Unexpected baseline frontend at {needle!r}")
    return source.replace(needle, replacement)


def sources():
    chat = (WEB / "modules/chat.js").read_text(encoding="utf-8")
    chat = once(chat, "decision: () => chatDecision, revealQuestion:",
                "revealOwnTask: id => historyReveal.task(id, isInstanceVisible), decision: () => chatDecision, revealQuestion:")
    pill = (WEB / "modules/main_waiting.js").read_text(encoding="utf-8")
    pill = once(pill, "revealQuestion = () => {}, reading =", "revealOwnTask, revealQuestion = () => {}, reading =")
    pill = once(pill, "    function targets() {", (HERE / "own_targets.js").read_text(encoding="utf-8") + "\n    function targets() {")
    pill = once(pill, "        return list.sort(", "        list.push(...ownTargets);\n        return list.sort(")
    pill = once(pill, "    function nodeOf(target) {", "    function nodeOf(target) {\n        if (target.kind === 'own') return ownResume(target);")
    pill = once(pill, "        let node = nodeOf(target);", """        if (target.kind === 'own') {
            await revealOwn(target); requestAnimationFrame(update); return;
        }
        let node = nodeOf(target);""")
    pill = once(pill, "            const actions = Array.isArray", "            ownTargets = resumeTargetsInMain(data);\n            const actions = Array.isArray")
    pill = once(pill, "            connected = value;", "            connected = value;\n            if (!value) ownTargets = [];")
    return {"chat.js": chat, "main_waiting.js": pill}


def script_response(script):
    def handle(route):
        route.fulfill(content_type="text/javascript", body=script)
    return handle


def install_variant(page):
    for filename, script in sources().items():
        page.route(f"**/static/modules/{filename}", script_response(script))
