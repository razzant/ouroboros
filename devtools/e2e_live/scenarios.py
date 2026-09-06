"""The scenario table of the live E2E stand: ``{id, prompt, settings_overrides, acceptance}``.

Every acceptance is a CALLABLE over durable artifacts (task_results rows, the lane clone's
git history, the task-drive ledgers, /proc) — never a keyword judgement of model prose
(BIBLE P5). The prompts are what a paid model sees; the ``stub_script`` of each row is the
$0 rehearsal of the same flow against the loopback stub model (``--stub``).

Owns its own reason to change (the product surface each scenario drives), so it lives
apart from the orchestration in ``run_live_lanes.py``.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import re
import subprocess
import tempfile
import time
from typing import Any, Callable

from devtools.benchmarks.common.server_runner import _api, _api_status
from devtools.e2e_live.ui_probe import GuardedUI

SM1_NEW_ACCENT = "#2f7de1"
SM1_COMMIT_MESSAGE = "ui: e2e_live SM1 accent token change (reviewed commit)"
SM1_CSS_PATH = "web/style.css"
# ``web/onboarding.css`` is inlined into the standalone first-run page and mirrors the app's
# ``:root`` tokens BY VALUE; ``tests/test_web_typography_static.py`` pins that every token both
# files declare resolves to the same value. ``--accent`` is one of them, so the change lands in
# BOTH files in one reviewed commit (the first paid run edited style.css alone and the tests
# preflight of ``commit_reviewed`` refused the commit on the parity invariant).
SM1_MIRROR_CSS_PATH = "web/onboarding.css"
SM1_CSS_PATHS = (SM1_CSS_PATH, SM1_MIRROR_CSS_PATH)
# The stub's README Version History row (the release preflight requires one per VERSION).
SM1_HISTORY_ROW = "Live E2E stand SM1 rehearsal: the brand accent changed in both stylesheets."
_ACCENT_RE = re.compile(r"^(\s*--accent:\s*)([^;]+);", re.MULTILINE)
_CSS_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_CSS_TOKEN_RE = re.compile(r"(--[a-z0-9-]+)\s*:\s*([^;]+);")
# ``release_sync.PRE_SUFFIX`` split into the parts the stub bump increments.
_VERSION_PARTS_RE = re.compile(r"^(\d+\.\d+\.)(\d+)(-?(?:rc|alpha|beta|a|b)\.?)?(\d+)?$", re.IGNORECASE)
# The runtime's typed refusal prefix on a blocked review tool result ("⚠️ CODE: ...").
_REFUSAL_CODE_RE = re.compile(r"⚠️\s*([A-Z][A-Z_]+):")
_REVIEW_TOOLS = ("preflight_review", "commit_reviewed")
# The prefix ``claude_advisory_review`` stamps on an oversize-prompt skip row's ``raw_result``.
_ADVISORY_SKIP_PREFIX = "⚠️ ADVISORY_SKIPPED:"
# The product's vision/browser inspection surfaces (``ouroboros/tools/vision.py``, ``browser.py``).
_VISION_TOOLS = ("analyze_screenshot", "vlm_query", "view_image")
_BROWSER_TOOL = "browser_action"

SW1_OBJECTIVE = (
    "E2E_LIVE_SW1: survey this repository with TWO scouts running in parallel. Delegate "
    "'list the top-level directories and name the three largest' to scout A and 'list the "
    "test modules under tests/system_e2e and count them' to scout B via schedule_subagent "
    "(subagent_id 'scout'), wait for both with wait_tasks, then summarize both results."
)
SW1_ROSTER_ID = "scout"

SK1_SKILL = "e2e_live_probe"
# The web owner's Main chat. The plugin below hard-binds every relayed line to it: the chat is
# never chosen by the caller, so the grant covers exactly one owner-facing destination.
SK1_OWNER_CHAT_ID = 1
SK1_ECHO_MAX_CHARS = 200
# The fixture is HONEST: every declared permission is exercised by ``SK1_PLUGIN`` and the prose
# names the narrow purpose of the privileged one. The first paid run declared ``inject_chat``
# over a plugin that never performed it (and prose denying host access), and the skill review
# refused it 3/3 on exactly ``permissions_honesty`` + ``inject_chat_minimization`` — the
# reviewers were right. ``tool`` = ``register_tool``; ``net`` = the ONE loopback HTTP request
# per call to the Host Service; ``inject_chat`` = that request's route.
SK1_SKILL_MD = f"""---
name: {SK1_SKILL}
description: Loopback probe extension authored by the live E2E stand SK1 scenario.
version: 0.1.0
type: extension
entry: plugin.py
plugin_api: "2.0"
permissions: ["tool", "inject_chat", "net"]
model_experience:
  what_model_sees: 'E2E_LIVE_SK1 adds a loopback probe echo tool that relays its reply into the owner chat'
  token_effect: 'one catalogue line'
---
Probe extension body: one `echo` tool. Each explicit call relays its own reply, `echo: <message>`,
as ONE bounded line (at most {SK1_ECHO_MAX_CHARS} characters) into the owner's own Main chat
through the loopback Host Service `/chat/inject` route, authenticated with this skill's token
(`get_skill_token().use_in_request()` at the request site only, never logged) and attributed by
the host as `skill:{SK1_SKILL}`. That is the whole purpose of `inject_chat`: a user-facing echo
the owner can see in their chat. The destination is fixed to the web owner's chat
(chat_id {SK1_OWNER_CHAT_ID}) and is never chosen by the caller; the sender is left unidentified
(no owner impersonation); there is no inbound listener, no polling, no retry, no broadcast, and
no host contacted other than 127.0.0.1. `net` covers exactly that one loopback request per call.
"""
SK1_PLUGIN = (
    "import json\n"
    "import os\n"
    "import urllib.parse\n"
    "import urllib.request\n"
    "\n"
    "LOOPBACK_HOSTS = ('127.0.0.1', 'localhost', '::1')\n"
    "\n"
    "\n"
    "def _loopback_base(raw):\n"
    "    # The host token is only ever presented to the loopback Host Service: a base URL\n"
    "    # that names any other host (or scheme) is refused before a request is built.\n"
    "    parts = urllib.parse.urlsplit(raw)\n"
    "    if parts.scheme != 'http' or parts.hostname not in LOOPBACK_HOSTS or parts.path not in ('', '/'):\n"
    "        raise RuntimeError('Host Service base must be a loopback http URL')\n"
    "    return raw.rstrip('/')\n"
    f"OWNER_CHAT_ID = {SK1_OWNER_CHAT_ID}   # the web owner's Main chat; never a caller-chosen chat\n"
    f"MAX_CHARS = {SK1_ECHO_MAX_CHARS}\n"
    "\n"
    "\n"
    "def register(api):\n"
    "    def _echo(ctx, message='hi'):\n"
    "        # One bounded line into the owner's own chat per explicit call, then the same\n"
    "        # text back to the caller. Loopback Host Service only; proxies from the\n"
    "        # environment are ignored so the request can never leave this host.\n"
    "        # ONE line: line breaks collapse to spaces; the cap applies to the FINAL text.\n"
    "        text = ('echo: ' + ' '.join(str(message).split()))[:MAX_CHARS]\n"
    "        base = _loopback_base(os.environ.get('HOST_SERVICE_URL') or (\n"
    "            'http://127.0.0.1:' + os.environ.get('OUROBOROS_HOST_SERVICE_PORT', '8767')))\n"
    "        request = urllib.request.Request(\n"
    "            base + '/chat/inject', method='POST',\n"
    f"            data=json.dumps({{'text': text, 'chat_id': OWNER_CHAT_ID, 'sender_label': '{SK1_SKILL}'}}).encode('utf-8'),\n"
    "            headers={'Content-Type': 'application/json',\n"
    "                     'X-Skill-Token': api.get_skill_token().use_in_request()},\n"
    "        )\n"
    "        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))\n"
    "        with opener.open(request, timeout=10) as response:\n"
    "            if response.status != 202:\n"
    "                raise RuntimeError(f'Host Service inject refused: HTTP {response.status}')\n"
    "        return text\n"
    "\n"
    "    api.register_tool(\n"
    "        'echo', _echo, description='echo probe: relays the reply into the owner chat',\n"
    "        schema={'type': 'object', 'properties': {'message': {'type': 'string'}}},\n"
    "    )\n"
)
# The ONLY privileged (owner-granted) permission the stand ever issues: the manifest above
# requests exactly it and the plugin performs exactly it, so "no other host grant" holds by
# construction of the grant call, not by a denylist (``tool``/``net`` are manifest declarations
# the loader enforces, not owner grants — ``skill_loader.requested_skill_permissions``).
SK1_GRANTS = ["inject_chat"]
SK1_ECHO_MESSAGE = "ping-e2e-live"
SK1_ECHO_EXPECTED = f"echo: {SK1_ECHO_MESSAGE}"   # exactly what ``_echo`` in SK1_PLUGIN returns AND relays


def _git(args: list[str], cwd: pathlib.Path) -> str:
    proc = subprocess.run(["git", *args], cwd=str(cwd), check=False, capture_output=True, text=True)
    return (proc.stdout or "").strip()


def accent_value(css_text: str) -> str:
    match = _ACCENT_RE.search(css_text)
    return match.group(2).strip() if match else ""


def css_with_accent(css_text: str, value: str) -> str:
    return _ACCENT_RE.sub(lambda m: f"{m.group(1)}{value};", css_text, count=1)


def css_root_tokens(css_text: str) -> dict[str, str]:
    """``{token: value}`` of the FIRST ``:root`` block, comments stripped — the same reading
    ``tests/test_web_typography_static.py`` applies to both stylesheets."""
    css = _CSS_COMMENT_RE.sub("", css_text)
    end = css.find("\n}")
    root = css if end < 0 else css[:end]
    if not root.lstrip().startswith(":root"):
        return {}
    return {name: " ".join(value.split()) for name, value in _CSS_TOKEN_RE.findall(root)}


def css_mirror_drift(style_css: str, onboarding_css: str) -> dict[str, tuple[str, str]]:
    """Shared ``:root`` tokens whose values differ between the two files (empty == parity)."""
    style, onboarding = css_root_tokens(style_css), css_root_tokens(onboarding_css)
    return {token: (style[token], onboarding[token])
            for token in sorted(set(style) & set(onboarding)) if style[token] != onboarding[token]}


def commit_refusal_facts(ledger: dict, tools_rows: list, stored: dict) -> dict:
    """The TYPED trail of every ``commit_reviewed``/``preflight_review`` refusal of a task.

    Three durable sources, none of them model prose: the advisory ledger's attempt rows
    (``phase``/``status``/``block_reason``) and advisory-run statuses, the tools.jsonl rows of
    the two review tools (their typed ``status`` plus the runtime's own ``⚠️ CODE:`` refusal
    prefix — PREFLIGHT_BLOCKED, TESTS_PREFLIGHT_BLOCKED, SCOPE_REVIEW_BLOCKED, ...), and the
    task's terminal ``reason_code`` (``budget_exhausted`` = BudgetExceeded, ``deadline_local``
    = the deadline). The first paid run's SM1 lanes failed on exactly this ladder and the
    result rows named none of it."""
    attempts = [a for a in (ledger.get("attempts") or []) if isinstance(a, dict)]
    runs = [r for r in (ledger.get("advisory_runs") or []) if isinstance(r, dict)]
    calls = []
    for row in tools_rows:
        if str(row.get("tool") or "") not in _REVIEW_TOOLS:
            continue
        match = _REFUSAL_CODE_RE.search(str(row.get("result_preview") or ""))
        calls.append({"tool": str(row.get("tool") or ""), "status": str(row.get("status") or ""),
                      "code": match.group(1) if match else ""})
    return {
        "commit_attempts": [{"attempt": a.get("attempt"), "phase": str(a.get("phase") or ""),
                             "status": str(a.get("status") or ""), "block_reason": str(a.get("block_reason") or "")}
                            for a in attempts],
        "advisory_run_statuses": [str(r.get("status") or "") for r in runs],
        "review_tool_calls": calls,
        "refusal_codes": sorted({c["code"] for c in calls if c["code"]}),
        "terminal_status": str(stored.get("status") or ""),
        "terminal_reason_code": str(stored.get("reason_code") or ""),
    }


def dispatch_verdict(rows: list, expected_text: str) -> dict:
    """What the durable tools.jsonl rows of an extension surface prove about its dispatch.

    ``extension_generation`` alone is NOT proof of a successful physical call: the dispatcher
    stamps it on failed outcomes too. A dispatch counts only when the row's typed ``status`` is
    ``ok`` AND the recorded result is exactly the extension's own output."""
    last = rows[-1] if rows else {}
    meta = last.get("tool_result_meta") if isinstance(last.get("tool_result_meta"), dict) else {}
    digest = str(meta.get("extension_generation") or "")
    return {"row_present": bool(rows), "status": str(last.get("status") or ""),
            "generation": digest, "generation_ok": bool(re.fullmatch(r"[0-9a-f]{8,64}", digest)),
            "physical_dispatch": meta.get("physical_dispatch") is True,
            "echo_ok": str(last.get("result_preview") or "").strip() == expected_text}


def owner_chat_relay_rows(chat_rows: list, skill: str, text: str) -> list:
    """The durable ``chat.jsonl`` rows proving the probe's relay reached the owner's chat: inbound
    rows the HOST stamped ``source=skill:<name>`` (attribution by the host, never by the plugin),
    filed under the owner's chat, carrying exactly the echo text."""
    return [row for row in chat_rows
            if str(row.get("direction") or "") == "in" and str(row.get("source") or "") == f"skill:{skill}"
            and int(row.get("chat_id") or 0) == SK1_OWNER_CHAT_ID and str(row.get("text") or "").strip() == text]


class DuplicateCheckKey(RuntimeError):
    """A scenario wrote the same check key twice (see ``LaneContext.check``)."""


class LaneContext:
    """What one lane hands its scenario: the live server, its clone/data root, the durable
    readers, the lazily opened UI client, and the two verdict maps the acceptance fills.

    ``ui_resolver(base_url)`` answers ``(open client, "")`` or ``(None, "ui_unavailable:<why>")``
    (``ui_probe.resolve_ui_client``); ``ui_reason`` non-empty at construction is the lane-start
    availability verdict and the client is never attempted. The browser opens on the FIRST
    ``ctx.ui`` use — never at lane start, so a task-long wait cannot kill it unseen — and a
    later Playwright failure degrades it to a typed reason instead of a lane crash."""

    def __init__(self, *, server: Any, clone: pathlib.Path, data_root: pathlib.Path, oracle: Any,
                 harness: Any, ui_resolver: Callable[[str], tuple[Any, str]] | None, ui_reason: str,
                 shots: pathlib.Path, log: Callable[[str], None], task_timeout: float,
                 restart: Callable[[], Any]) -> None:
        self.server = server
        self.clone = pathlib.Path(clone)
        self.data_root = pathlib.Path(data_root)
        self.oracle = oracle
        self.h = harness  # tests.system_e2e.harness: wait_until / wait_durable_result / proc oracles
        self._ui_resolver = ui_resolver
        self._ui: GuardedUI | None = None
        self._ui_probe_reason = ui_reason  # the lane-start availability verdict: permanent
        self.ui_reason = ui_reason
        self.shots = pathlib.Path(shots)
        self.log = log
        self.task_timeout = float(task_timeout)
        self._restart = restart
        self.checks: dict[str, bool] = {}
        self.facts: dict[str, Any] = {}
        self.screenshots: list[str] = []

    def check(self, name: str, ok: bool, **facts: Any) -> bool:
        """One verdict per key. A second write to the same key is a scenario bug (the SK1
        author/dispatch awaits once shared ``http_terminal_completed`` and the later one
        erased the earlier), so it is refused loudly instead of silently winning."""
        if name in self.checks:
            raise DuplicateCheckKey(f"check {name!r} already recorded for this lane; namespace it per task")
        self.checks[name] = bool(ok)
        self.facts.update(facts)
        return bool(ok)

    def submit(self, description: str, *, metadata: dict | None = None) -> str:
        body = {
            "description": description, "memory_mode": "forked", "actor_id": "e2e_live",
            "source": "e2e_live", "timeout_sec": int(self.task_timeout),
            "metadata": {"source": "e2e_live", "delegation_role": "root", **(metadata or {})},
        }
        created = _api(self.server.base_url, "POST", "/api/tasks", body, timeout=60)
        task_id = str(created.get("task_id") or "")
        if not task_id:
            raise RuntimeError(f"task submit refused: {created!r}")
        return task_id

    def wait_task(self, task_id: str, *, label: str = "") -> dict:
        """Wait for the HTTP terminal, then for the DURABLE terminal row (they differ in time).

        ``label`` prefixes the two check keys (``author_http_terminal_completed``, ...) so a
        scenario awaiting several tasks keeps one verdict PER task instead of the last await
        overwriting the earlier ones."""
        prefix = f"{label}_" if label else ""
        result = self.server.wait_task(task_id, timeout=self.task_timeout + 300)
        if str(result.get("status") or "") == "timeout":
            self.server.cancel_task(task_id)
            result = self.server.wait_task(task_id, timeout=300)
        self.check(f"{prefix}http_terminal_completed", result.get("status") == "completed",
                   **{f"{prefix}http_status": str(result.get("status") or "")})
        stored = {}
        try:
            stored = self.h.wait_durable_result(self.oracle, task_id, timeout=180)
        except AssertionError as exc:
            self.facts[f"{prefix}durable_result_error"] = str(exc)[:500]
        self.check(f"{prefix}durable_terminal_completed", stored.get("status") == "completed")
        terminal = stored or result
        self.facts[f"{prefix}terminal"] = {"task_id": task_id, "status": str(terminal.get("status") or ""),
                                           "reason_code": str(terminal.get("reason_code") or "")}
        self.facts["runtime_result"] = terminal  # the lane's runtime disclosure: the LAST awaited task
        return terminal

    def wait_events(self, oracle: Any, event_type: str, predicate: Callable[[dict], bool], timeout: float = 90) -> list:
        """Rows of ``event_type`` matching ``predicate``, waiting for the ASYNC event queue: the
        durable task row lands before ``events.jsonl`` catches up, so a read right after the
        terminal would race the writer."""
        return self.h.wait_until(
            lambda: [row for row in oracle.events(event_type) if predicate(row)] or None, timeout) or []

    def check_paid_tokens(self, task_ids: list[str]) -> None:
        """NOT fail-open: a 0/0 llm_usage row is the crashed-subprocess / silent-403 signature."""
        ids = set(task_ids)
        rows = self.wait_events(self.oracle, "llm_usage",
                                lambda row: str(row.get("task_id") or "") in ids or str(row.get("root_task_id") or "") in ids)
        prompt_tokens = sum(int(row.get("prompt_tokens") or 0) for row in rows)
        self.check("prompt_tokens_positive", prompt_tokens > 0,
                   llm_usage_rows=len(rows), prompt_tokens=prompt_tokens,
                   completion_tokens=sum(int(row.get("completion_tokens") or 0) for row in rows))

    @property
    def ui(self) -> GuardedUI | None:
        """The guarded UI client, opened on first use against the CURRENT server; ``None`` with
        ``ui_reason`` typed when it cannot be opened. Once opened it stays the answer even after
        a failure (its calls become no-ops, ``ui_reason`` names the failure), so a scenario that
        tested ``ctx.ui is not None`` before the failure keeps running to its own checks."""
        if self._ui is None and not self.ui_reason and self._ui_resolver is not None:
            client, reason = self._ui_resolver(self.server.base_url)
            if client is None:
                self.ui_reason = reason
            else:
                self._ui = GuardedUI(client, self._ui_unavailable)
        return self._ui

    def _ui_unavailable(self, reason: str, error: str) -> None:
        self.ui_reason = reason
        self.facts["ui_reason"] = reason
        self.facts.setdefault("ui_errors", []).append(error)
        self.log(f"ui unavailable: {error}")

    def close_ui(self) -> None:
        """Close the open probe if any (errors ignored: the browser may already be dead)."""
        if self._ui is not None:
            self._ui.close()
            self._ui = None

    def screenshot(self, name: str) -> None:
        ui = self.ui
        if ui is None:
            return
        path = self.shots / f"{name}.png"
        ui.screenshot(path)
        if not self.ui_reason:  # the shot is evidence, never a gate: a failed one is typed in ui_reason
            self.screenshots.append(str(path))

    def restart(self) -> None:
        """Restart the server, then drop the probe: the self-mod re-exec keeps the port but a
        non-self-mod restart may not, and a browser that died during the wait must not be reused.
        The next ``ctx.ui`` use opens a fresh client against the restarted server's base_url."""
        self.server = self._restart()
        self.close_ui()
        self.ui_reason = self._ui_probe_reason


# --------------------------------------------------------------------------- #
# SM1 — the brand accent changes and lands as a reviewed RELEASE (the S2 set + UI)
# --------------------------------------------------------------------------- #
# The realistic design: an owner asks for a new brand accent, and the product's own policy
# shapes the work — the design system (docs/DESIGN.md §3: one ``--accent`` value, its named roles
# and the accent alpha ladder), the visible-UI evidence rule (a real consumer flow inspected in
# the browser), and BIBLE P9 (every commit is a release: the synchronized carriers bump in the
# same diff). The stand describes that task and pins only the observable contract; it never
# routes around the review organs. Incident (first paid run on v7.0.0-rc.14, three attempts):
# the earlier prompt changed ONLY ``--accent``, forbade the bump and skipped the advisory
# preflight, and the commit gate refused it three ways — [triad] version_bump (CHECKLISTS item 8
# / P9), [scope] a mixed blue/red palette against DESIGN §3 (the red-derived roles and ladder
# were left behind), [triad] development_compliance 2(i) (no vision-inspected UI evidence).

def sm1_prompt() -> str:
    return (
        f"Change the product's brand accent to {SM1_NEW_ACCENT} (a blue) consistently with the design "
        "system (docs/DESIGN.md §3: the one --accent value, its named roles and the accent alpha "
        f"ladder) in BOTH {SM1_CSS_PATH} and {SM1_MIRROR_CSS_PATH}, keeping the two sheets' token "
        "parity (tests/test_web_typography_static.py pins it); verify the visible result the way the "
        "review policy requires (exercise at least one real consumer flow in the browser and inspect "
        "it); then land it as a reviewed release through preflight_review and commit_reviewed with "
        f"commit message '{SM1_COMMIT_MESSAGE}', following the release policy the review organs "
        "enforce (every commit is a release: the synchronized version carriers are bumped in the same "
        "diff — a patch bump). Finish once the commit has landed."
    )


def version_is_bumped(before: str, after: str) -> bool:
    """``after`` is a release version in the product's grammar (``release_sync``) AND a valid
    PEP 440 version strictly greater than ``before`` (``packaging`` ships with pytest, so it is
    present wherever the stand or its pins run)."""
    from packaging.version import InvalidVersion, Version

    from ouroboros.tools.release_sync import is_release_version

    try:
        return is_release_version(after) and Version(after.strip()) > Version(before.strip())
    except InvalidVersion:
        return False


def advisory_run_is_real(run: dict) -> bool:
    """A ledger row the advisory reviewer actually produced. ``fresh`` is written only by a
    completed reviewer episode; ``stale`` is ANY aged row (fresh, bypassed or skipped), told
    apart by the bypass fields and the skip prefix the two audited paths stamp."""
    status = str(run.get("status") or "")
    if status == "fresh":
        return True
    return (status == "stale" and not run.get("bypass_reason") and not run.get("bypassed_by_task")
            and not str(run.get("raw_result") or "").startswith(_ADVISORY_SKIP_PREFIX))


def vision_evidence_rows(tools_rows: list) -> list:
    """tools.jsonl rows of a browser/vision inspection: the vision tools, or a browser screenshot."""
    out = []
    for row in tools_rows:
        tool = str(row.get("tool") or "")
        args = row.get("args") if isinstance(row.get("args"), dict) else {}
        if tool in _VISION_TOOLS or (tool == _BROWSER_TOOL and str(args.get("action") or "") == "screenshot"):
            out.append(row)
    return out


def _git_show(clone: pathlib.Path, rev: str, path: str) -> str:
    """The exact text of ``path`` at ``rev`` ('' when absent there)."""
    proc = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=str(clone), check=False, capture_output=True, text=True)
    return proc.stdout if proc.returncode == 0 else ""


def release_carriers_desync_at(clone: pathlib.Path, rev: str) -> str:
    """The product's own release-metadata admission gate, read over the carrier files of
    ``rev`` exported to a scratch root (the commit, not the worktree): '' when VERSION, its
    README row and every carrier the SSOT names agree, else the gate's PREFLIGHT_BLOCKED text."""
    from ouroboros.commit_admission import release_metadata_preflight
    from ouroboros.tools.release_sync import CARRIER_SPAN_PATHS

    with tempfile.TemporaryDirectory(prefix="sm1_carriers_") as tmp:
        root = pathlib.Path(tmp)
        for rel in sorted(CARRIER_SPAN_PATHS):
            text = _git_show(clone, rev, rel)
            if text:
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).write_text(text, encoding="utf-8")
        return str(release_metadata_preflight(root, SM1_COMMIT_MESSAGE, ["VERSION"]) or "")


def sm1_out_of_scope(clone: pathlib.Path, rev: str, files: list[str]) -> list[str]:
    """Committed paths OUTSIDE the SM1 contract: the two stylesheets, the release carriers the
    SSOT names, ``docs/DESIGN.md``, and any other ``web/**/*.css`` whose change is comment-only."""
    from ouroboros.tools.release_sync import CARRIER_SPAN_PATHS

    def comment_only(path: str) -> bool:
        return (_CSS_COMMENT_RE.sub("", _git_show(clone, f"{rev}^", path))
                == _CSS_COMMENT_RE.sub("", _git_show(clone, rev, path)))

    return [path for path in files
            if path not in SM1_CSS_PATHS and path not in CARRIER_SPAN_PATHS and path != "docs/DESIGN.md"
            and not (path.startswith("web/") and path.endswith(".css") and comment_only(path))]


def run_sm1(ctx: LaneContext) -> None:
    before = {path: (ctx.clone / path).read_text(encoding="utf-8") for path in SM1_CSS_PATHS}
    version_before = (ctx.clone / "VERSION").read_text(encoding="utf-8").strip()
    ctx.facts["accent_before"] = accent_value(before[SM1_CSS_PATH])
    ctx.facts["version_before"] = version_before
    task_id = ctx.submit(sm1_prompt())
    ctx.facts["task_id"] = task_id
    stored = ctx.wait_task(task_id)
    # The commit LANDED in the lane clone: under blocking enforcement that is only reachable
    # through PASS verdicts from both review organs. Located by its message (the post-task
    # evolve cycle may commit after it), read from that commit, never from the worktree.
    commit = next((sha for sha, _sep, subject in (line.partition("\x00") for line in _git(
        ["log", "-n", "5", "--format=%H%x00%s"], ctx.clone).splitlines()) if SM1_COMMIT_MESSAGE in subject), "")
    ctx.check("commit_landed", bool(commit), commit_sha=commit)
    rev = commit or "HEAD"
    committed = {path: _git_show(ctx.clone, rev, path) for path in SM1_CSS_PATHS}
    ctx.check("committed_css_carries_new_accent",
              all(accent_value(committed[p]) == SM1_NEW_ACCENT and committed[p] != before[p] for p in SM1_CSS_PATHS),
              accent_committed={p: accent_value(committed[p]) for p in SM1_CSS_PATHS})
    # The WHOLE shared ``:root`` token set agrees between the sheets: a re-derived accent family
    # is fine exactly as long as both carry the same values.
    drift = css_mirror_drift(committed[SM1_CSS_PATH], committed[SM1_MIRROR_CSS_PATH])
    ctx.check("committed_css_mirror_parity", not drift, css_mirror_drift=drift)
    version_after = _git_show(ctx.clone, rev, "VERSION").strip()
    ctx.check("committed_version_bumped", version_is_bumped(version_before, version_after), version_committed=version_after)
    desync = release_carriers_desync_at(ctx.clone, rev)
    ctx.check("committed_release_carriers_in_sync", desync == "", release_carriers_desync=desync[:500])
    files = [f for f in _git(["show", "--format=", "--name-only", rev], ctx.clone).splitlines() if f]
    # The stand pins only the observable contract: both sheets are IN the commit. Files beyond
    # the sheets, the carriers and the documented companions are recorded as a fact, never a
    # failure: a scope reviewer may legitimately name another accent touchpoint (an inline
    # colour on the unlock page, the site stylesheet), and the reviewers own that judgment.
    ctx.check("committed_diff_includes_sheets", all(path in files for path in SM1_CSS_PATHS),
              committed_files=files, committed_companions=sm1_out_of_scope(ctx.clone, rev, files))
    ctx.check("worktree_clean_after_commit", _git(["status", "--porcelain"], ctx.clone) == "")
    task_oracle = ctx.oracle.task_drive(task_id)
    ledger = task_oracle.advisory_review()
    runs = [r for r in (ledger.get("advisory_runs") or []) if isinstance(r, dict)]
    # A REAL advisory run, not the audited skip/bypass row the earlier prompt routed through.
    ctx.check("advisory_ledger_row_present", any(advisory_run_is_real(r) for r in runs))
    tools_rows = task_oracle.tools_rows()
    ctx.facts["commit_reviewed_refusals"] = commit_refusal_facts(ledger, tools_rows, stored)
    # A FACT, not a check: the reviewers judge the UI evidence (development_compliance 2(i)).
    vision = vision_evidence_rows(tools_rows)
    ctx.facts["vision_evidence_present"] = bool(vision)
    ctx.facts["vision_evidence_tools"] = sorted({str(r.get("tool") or "") for r in vision})
    ctx.check("scope_review_complete_event",
              bool(ctx.wait_events(task_oracle, "scope_review_complete", lambda _row: True)))
    ctx.check_paid_tokens([task_id])
    # R12: the computed style is read from the COMMITTED CSS after a restart.
    ctx.restart()
    if ctx.ui is None:
        ctx.check("ui_computed_style", False, ui_reason=ctx.ui_reason)
        return
    ctx.ui.goto("/")
    observed = str(ctx.ui.computed_property(":root", "--accent") or "").strip()
    # Only meaningful on a landed commit: a served working-tree edit would show the same value.
    ctx.check("ui_computed_style", observed == SM1_NEW_ACCENT and ctx.checks["commit_landed"],
              accent_computed=observed)
    ctx.screenshot("sm1_after_restart")


def sm1_next_version(version: str, taken: "frozenset[str] | set[str]" = frozenset()) -> str:
    """The smallest strictly-greater release version the stub bumps to: the next pre-release
    number on a pre-release seed (``7.0.0-rc.14`` -> ``7.0.0-rc.15``), else the next patch —
    skipping every version whose tag ``v<version>`` is already in ``taken``: a seed cloned from
    an older ref carries the newer tags, and the review binding refuses a staged version whose
    tag exists (``git_review_cycle._prepare_review_binding``)."""
    candidate = version.strip()
    for _ in range(1000):
        match = _VERSION_PARTS_RE.match(candidate)
        if match is None:
            raise ValueError(f"seed VERSION is not a release version: {candidate!r}")
        base, patch, pre, number = match.groups()
        candidate = f"{base}{patch}{pre}{int(number) + 1}" if pre else f"{base}{int(patch) + 1}"
        if f"v{candidate}" not in taken:
            return candidate
    raise ValueError(f"no free release version above {version.strip()!r}")


def readme_with_history_row(readme: str, version: str, description: str) -> str:
    """README with ONE new Version History row for ``version`` on top of the table, trimmed to
    the P9 row limits the release preflight enforces: while ``check_history_limit`` complains,
    the oldest row whose removal reduces the complaints goes — the trim the gate text asks for."""
    from ouroboros.tools.release_sync import _VERSION_ROW_RE, check_history_limit

    first = _VERSION_ROW_RE.search(readme)
    if first is None:
        raise ValueError("README has no Version History rows")
    row = f"| {version} | {time.strftime('%Y-%m-%d', time.gmtime())} | {description} |\n"
    text = readme[:first.start()] + row + readme[first.start():]
    while check_history_limit(text):
        for match in reversed(list(_VERSION_ROW_RE.finditer(text))):
            end = text.find("\n", match.start())
            candidate = text[:match.start()] + ("" if end < 0 else text[end + 1:])
            if len(check_history_limit(candidate)) < len(check_history_limit(text)):
                text = candidate
                break
        else:
            raise ValueError("cannot trim the Version History to the P9 limits")
    return text


def sm1_release_writes(clone: pathlib.Path, version: str) -> list[dict]:
    """The bump's carrier edits, computed OFFLINE through the product's release-sync SSOT: the
    clone's carrier files (every path ``CARRIER_SPAN_PATHS`` names) are copied to a scratch root,
    VERSION is bumped, the README gains its Version History row (changelog prose stays manual
    by the SSOT's own contract), ``sync_release_metadata`` rewrites the rest, and each file that
    differs from the clone becomes one ``write_file`` step. No hand-written carrier list: the
    stub's tool vocabulary has no release-sync tool, so the projection is built here."""
    from ouroboros.tools.release_sync import CARRIER_SPAN_PATHS, sync_release_metadata

    with tempfile.TemporaryDirectory(prefix="sm1_release_") as tmp:
        root = pathlib.Path(tmp)
        before = {}
        for rel in sorted(CARRIER_SPAN_PATHS):
            if (clone / rel).is_file():
                before[rel] = (clone / rel).read_text(encoding="utf-8")
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).write_text(before[rel], encoding="utf-8")
        (root / "VERSION").write_text(f"{version}\n", encoding="utf-8")
        (root / "README.md").write_text(
            readme_with_history_row(before["README.md"], version, SM1_HISTORY_ROW), encoding="utf-8")
        sync_release_metadata(str(root))
        after = {rel: (root / rel).read_text(encoding="utf-8") for rel in before}
    return [{"tool": "write_file", "arguments": {"root": "system_repo", "path": rel, "content": after[rel]}}
            for rel in before if after[rel] != before[rel]]


def sm1_stub_script(clone: pathlib.Path) -> dict:
    # The mock reviewers do not judge design, so the stub changes the one value; the paid
    # model re-derives the roles and the ladder under the real reviewers.
    writes = [{"tool": "write_file", "arguments": {
        "root": "system_repo", "path": path,
        "content": css_with_accent((clone / path).read_text(encoding="utf-8"), SM1_NEW_ACCENT)}}
        for path in SM1_CSS_PATHS]
    taken = frozenset(_git(["tag", "-l"], clone).split())
    writes.extend(sm1_release_writes(clone, sm1_next_version((clone / "VERSION").read_text(encoding="utf-8"), taken)))
    return {"agent": [
        *writes,
        # The full user path, no skip flags: the release preflight sees VERSION in scope, the
        # advisory episode runs against the stub (a REAL ledger row), and the hermetic suite is
        # the tests preflight exactly like the paid prompt (``preflight_runner._preflight_env``
        # scrubs every settings key the loopback lane projects).
        {"tool": "preflight_review", "arguments": {"commit_message": SM1_COMMIT_MESSAGE}},
        {"tool": "commit_reviewed", "arguments": {
            "commit_message": SM1_COMMIT_MESSAGE, "paths": [w["arguments"]["path"] for w in writes],
            "goal": "Change the brand accent for the live E2E stand and release it",
            "scope": f"{SM1_CSS_PATH}, {SM1_MIRROR_CSS_PATH} and the release version carriers."}},
        {"final": "SM1 done: the brand accent change landed as a reviewed release."},
    ]}


# --------------------------------------------------------------------------- #
# SW1 — Swarm: force_plan + roster, >=2 children, fanout receipt, cost rollup, no orphans
# --------------------------------------------------------------------------- #

def sw1_roster(child_model: str) -> str:
    return json.dumps({"enabled": True, "items": [{
        "subagent_id": SW1_ROSTER_ID,
        "recommended_use": "Read-only scout for parallel repository surveys.",
        "route": {"kind": "api_model", "target_id": child_model},
        "effort": "low",
    }]})


def _find_root_task(ctx: LaneContext, marker: str) -> str:
    results_dir = ctx.data_root / "task_results"
    for path in sorted(results_dir.glob("*.json")) if results_dir.is_dir() else []:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(row, dict) and not row.get("parent_task_id") and marker in json.dumps(row):
            return str(row.get("task_id") or path.stem)
    return ""


def run_sw1(ctx: LaneContext) -> None:
    ui_path = ctx.ui is not None
    if ui_path:
        # The owner's path: the Swarm button arms force_plan on the WS chat frame.
        ctx.ui.goto("/")
        ctx.ui.send_chat(SW1_OBJECTIVE, swarm=True)
        ctx.screenshot("sw1_swarm_sent")
        parent_id = ctx.h.wait_until(lambda: _find_root_task(ctx, "E2E_LIVE_SW1"), 300) or ""
    else:
        parent_id = ctx.submit(SW1_OBJECTIVE, metadata={"force_plan": True, "force_plan_source": "swarm"})
    ctx.check("ui_swarm_path_exercised", ui_path, ui_reason=ctx.ui_reason)
    ctx.check("root_task_admitted", bool(parent_id), task_id=parent_id)
    if not parent_id:
        return
    stored = ctx.wait_task(parent_id)
    ctx.check("plan_review_engaged", isinstance(stored.get("plan_review_state"), dict))
    # The quiescent path, not the forced one: every child result absorbed before the final.
    ctx.check("clean_finalization", str(stored.get("reason_code") or "") != "children_unabsorbed",
              parent_reason_code=str(stored.get("reason_code") or ""))
    children = ctx.oracle.child_task_ids(parent_id)
    ctx.check("at_least_two_children", len(children) >= 2, children=children)
    lineage_ok = bool(children)
    for child_id in children:
        row = ctx.oracle.task_result(child_id)
        provenance = row.get("depth_provenance") if isinstance(row.get("depth_provenance"), dict) else {}
        lineage_ok = lineage_ok and (
            row.get("parent_task_id") == parent_id and row.get("root_task_id") == parent_id
            and row.get("delegation_role") == "subagent" and int(provenance.get("achieved_depth") or 0) >= 1
            and row.get("status") == "completed")
    ctx.check("children_causal_lineage", lineage_ok)
    fanouts = ctx.oracle.task_drive(parent_id).events("swarm_fanout")
    fanned = {str(t) for row in fanouts for t in (row.get("task_ids") or [])}
    ctx.check("swarm_fanout_receipt_covers_children", bool(children) and set(children) <= fanned,
              fanout_task_ids=sorted(fanned))
    done = ctx.wait_events(ctx.oracle, "task_done", lambda row: str(row.get("task_id") or "") == parent_id)
    ctx.check("cost_rollup_with_children",
              bool(done) and "accounted_upper_bound_usd_with_children" in done[-1]
              and ctx.h.retired_cost_alias_paths(done[-1]) == [],
              accounted_upper_bound_usd_with_children=(done[-1].get("accounted_upper_bound_usd_with_children") if done else None))
    tree = set(ctx.h.process_tree_pids(ctx.server.proc.pid))
    carriers = ctx.h.pids_with_env_value(str(ctx.data_root))
    ctx.check("no_orphans_during_run", bool(carriers) and all(pid in tree for pid in carriers),
              env_carrier_pids=len(carriers))
    ctx.check_paid_tokens([parent_id, *children])
    ctx.screenshot("sw1_done")


def sw1_stub_script(_clone: pathlib.Path) -> dict:
    child_id_re = re.compile(r"Subagent request queued ([0-9a-f]{8})")
    # The wait_tasks projection pairs each child id with its exact result hash.
    child_result_re = re.compile(r'"task_id": "([0-9a-f]{8})".{0,400}?"child_result_sha256": "([0-9a-f]{64})"', re.DOTALL)

    def wait_step(text: str) -> dict:
        ids = sorted(set(child_id_re.findall(text)))
        if len(ids) < 2:
            return {"final": "E2E_SCRIPT_ERROR: fewer than two scheduled child ids visible"}
        return {"tool": "wait_tasks", "arguments": {"task_ids": ids, "timeout_sec": 300, "mode": "all_terminal"}}

    def dispose_step(index: int):
        def step(text: str) -> dict:
            pairs = sorted(set(child_result_re.findall(text)))
            if len(pairs) <= index:
                return {"final": "E2E_SCRIPT_ERROR: child result hash missing for the disposition"}
            child_id, sha = pairs[index]
            return {"tool": "tree_note", "arguments": {
                "kind": "decision", "text": f"Absorbed scout {child_id} into the summary.",
                "payload": {"type": "child_result_disposition", "child_task_id": child_id,
                            "disposition": "integrated", "child_result_sha256": sha}}}
        return step

    def scout(label: str, objective: str) -> dict:
        return {"tool": "schedule_subagent", "arguments": {
            "subagent_id": SW1_ROSTER_ID, "objective": f"Scout {label}: {objective}",
            "expected_output": "A short listing."}}

    return {
        "router": [{"tool": "promote_chat_to_task", "arguments": {
            "objective": SW1_OBJECTIVE, "title": "SW1 swarm survey", "predecessor_task_id": ""}},
            {"final": "Routed the Swarm request into a managed task."}],
        "agent": [
            {"tool": "plan_task", "arguments": {
                "goal": "Survey the repository with two parallel scouts.",
                "plan": "Schedule two scouts, wait for both, summarize.",
                "spec": {"deliverables": ["Two scout results summarized."],
                         "acceptance_claims": ["Both scouts completed and were absorbed."]}}},
            scout("A", "list the top-level directories"),
            scout("B", "list the test modules under tests/system_e2e"),
            wait_step,
            dispose_step(0),
            dispose_step(1),
            {"final": "SW1_PARENT_DONE: both scouts absorbed."},
        ],
        "child": [{"final": "SW1_CHILD_DONE: survey complete."}],
        "probe": [{"final": "No existing task duplicates this request."}],
    }


# --------------------------------------------------------------------------- #
# SK1 — the model authors a skill; the owner side reviews, grants, enables, dispatches
# --------------------------------------------------------------------------- #

def sk1_prompt() -> str:
    return (
        f"Author a new external skill named '{SK1_SKILL}' using write_file with root='skill_payload', "
        f"bucket='external', skill_name='{SK1_SKILL}'. Write exactly two files. SKILL.md:\n"
        f"{SK1_SKILL_MD}\nplugin.py:\n{SK1_PLUGIN}\nThen run skill_preflight(skill='{SK1_SKILL}') and "
        "finish; do not review, enable or grant anything yourself."
    )


def _skill_entry(base_url: str, name: str) -> dict:
    """The skill's ``/api/extensions`` row, or ``{}`` — a listing that fails is a failed check with facts
    (``review_executable``/``live_loaded`` absent), never an infra_error that aborts the lifecycle."""
    resp = _api_status(base_url, "GET", "/api/extensions", None, timeout=30)
    listing = resp["body"] if resp["status"] == 200 else {}
    rows = listing if isinstance(listing, list) else (listing.get("extensions") or listing.get("skills") or [])
    return next((row for row in rows if isinstance(row, dict) and row.get("name") == name), {})


def sk1_review_gate(review: dict, entry: dict, findings: list) -> tuple[bool, dict]:
    """The SK1 review criterion is the PRODUCT gate (owner decision 2026-09-06): the review ran (HTTP 200 with
    recorded findings) and the ``/api/extensions`` row says ``executable_review`` — clean, warnings, or blockers
    under advisory enforcement by operator choice (``skill_review_gate``). A clean all-PASS review is a recorded
    FACT, not the verdict: the rc.15 SK1 rerun on 560f7d71 authored one clean, one warnings and one blockers
    payload with every other lifecycle check green, so all-PASS measured the author model, not the product.
    The verdict also needs the review call itself to answer 200 with its own ``executable_review`` (the
    lifecycle's gate; a ``pending`` duplicate job never passes) and persisted findings (a review really ran).
    The SK1 lane sets no enforcement, so it runs the tree default (advisory today) under both profiles: the
    ``blockers under blocking enforcement`` branch is the product's rule, not a path the stand exercises."""
    gate = entry.get("review_gate") if isinstance(entry.get("review_gate"), dict) else {}
    failed = [f.get("item") for f in findings if str(f.get("verdict") or "") != "PASS"]
    ok = (review["status"] == 200 and review["body"].get("executable_review") is True
          and entry.get("executable_review") is True and bool(findings))
    return ok, {"review_status": review["body"].get("status"), "review_executable": entry.get("executable_review"),
                "review_body_executable": review["body"].get("executable_review"),
                "review_enforcement": gate.get("review_enforcement"), "review_blocking_reason": gate.get("blocking_reason"),
                "findings": len(findings), "findings_failed": failed, "review_clean": bool(findings) and not failed}


def run_sk1(ctx: LaneContext) -> None:
    from ouroboros.extension_surface_names import extension_surface_name

    payload_dir = ctx.data_root / "skills" / "external" / SK1_SKILL
    author_id = ctx.submit(sk1_prompt())
    ctx.facts["author_task_id"] = author_id
    ctx.wait_task(author_id, label="author")
    ctx.check("payload_authored_by_model", (payload_dir / "SKILL.md").is_file() and (payload_dir / "plugin.py").is_file())
    preflight_rows = [r for r in ctx.oracle.task_drive(author_id).tools_rows() if r.get("tool") == "skill_preflight"]
    ctx.check("skill_preflight_called", bool(preflight_rows))
    if not payload_dir.is_dir():
        return
    review = _api_status(ctx.server.base_url, "POST", f"/api/skills/{SK1_SKILL}/review", {}, timeout=900)
    review_state = ctx.oracle._json(f"state/skills/{SK1_SKILL}/review.json")
    findings = [f for f in (review_state.get("findings") or []) if isinstance(f, dict)]
    ok, review_facts = sk1_review_gate(review, _skill_entry(ctx.server.base_url, SK1_SKILL), findings)
    ctx.check("review_executable", ok, **review_facts)
    grants = _api_status(ctx.server.base_url, "POST", f"/api/skills/{SK1_SKILL}/grants", {"items": SK1_GRANTS}, timeout=120)
    granted = ctx.oracle._json(f"state/skills/{SK1_SKILL}/grants.json").get("granted_permissions")
    ctx.check("grants_exactly_requested", (grants["body"].get("grants") or {}).get("all_granted") is True
              and granted == SK1_GRANTS, granted_permissions=granted)
    toggled = _api_status(ctx.server.base_url, "POST", f"/api/skills/{SK1_SKILL}/toggle", {"enabled": True}, timeout=300)
    entry = _skill_entry(ctx.server.base_url, SK1_SKILL)
    ctx.check("enabled_live_loaded", toggled["body"].get("enabled") is True and not toggled["body"].get("error")
              and entry.get("live_loaded") is True and entry.get("dispatch_live") is True)
    ctx.screenshot("sk1_enabled")
    surface = extension_surface_name(SK1_SKILL, "echo")
    dispatch_id = ctx.submit(f"Call the tool `{surface}` once with message '{SK1_ECHO_MESSAGE}', then finish.")
    ctx.facts["dispatch_task_id"] = dispatch_id
    ctx.wait_task(dispatch_id, label="dispatch")
    rows = [r for r in ctx.oracle.task_drive(dispatch_id).tools_rows() if str(r.get("tool") or "") == surface]
    verdict = dispatch_verdict(rows, SK1_ECHO_EXPECTED)
    ctx.check("dispatch_durable_row_with_generation", verdict["row_present"] and verdict["generation_ok"],
              extension_generation=verdict["generation"])
    ctx.check("dispatch_physical_call_ok_with_echo", verdict["status"] == "ok" and verdict["echo_ok"],
              dispatch_status=verdict["status"], dispatch_echo_ok=verdict["echo_ok"],
              dispatch_physical=verdict["physical_dispatch"])
    # The granted permission was EXERCISED: every successful echo call put exactly one host-
    # attributed line into the owner's chat (the inbound row lands asynchronously, after the
    # Host Service's 202, so this waits on the durable chat log instead of reading it once).
    ok_calls = sum(1 for r in rows if str(r.get("status") or "") == "ok")
    relayed = ctx.h.wait_until(
        lambda: owner_chat_relay_rows(ctx.oracle._jsonl("logs/chat.jsonl"), SK1_SKILL, SK1_ECHO_EXPECTED) or None,
        90) or []
    ctx.check("dispatch_relayed_one_line_per_call_into_owner_chat", bool(relayed) and len(relayed) == ok_calls,
              owner_chat_relay_rows=len(relayed), dispatch_ok_calls=ok_calls)
    ctx.check_paid_tokens([author_id, dispatch_id])
    _api_status(ctx.server.base_url, "POST", f"/api/skills/{SK1_SKILL}/toggle", {"enabled": False}, timeout=300)
    deleted = _api_status(ctx.server.base_url, "POST", f"/api/skills/{SK1_SKILL}/delete", {}, timeout=120)
    ctx.check("deleted_payload_and_state", deleted["status"] == 200 and not deleted["body"].get("error")
              and not payload_dir.exists() and not (ctx.data_root / "state" / "skills" / SK1_SKILL).exists())


def sk1_stub_script(_clone: pathlib.Path) -> dict:
    from ouroboros.extension_surface_names import extension_surface_name

    payload = {"root": "skill_payload", "bucket": "external", "skill_name": SK1_SKILL}
    return {"agent": [
        {"tool": "write_file", "arguments": {**payload, "path": "SKILL.md", "content": SK1_SKILL_MD}},
        {"tool": "write_file", "arguments": {**payload, "path": "plugin.py", "content": SK1_PLUGIN}},
        {"tool": "skill_preflight", "arguments": {"skill": SK1_SKILL}},
        {"final": "SK1_AUTHORED: payload written and preflighted."},
        {"tool": extension_surface_name(SK1_SKILL, "echo"), "arguments": {"message": SK1_ECHO_MESSAGE}},
        {"final": "SK1_DISPATCH_DONE: echo absorbed."},
        # The relayed line opens ONE owner-chat turn on the same wire (tool-bearing, so it routes
        # as ``agent``); it and the dispatch task's closing round each take one of these finals.
        {"final": "SK1_OWNER_CHAT_LINE_SEEN: the probe relayed its echo; nothing to do."},
    ]}


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #

@dataclasses.dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    prompt: str
    settings_overrides: dict
    needs_ui: bool
    acceptance: Callable[[LaneContext], None]
    stub_script: Callable[[pathlib.Path], dict]
    # ROOT tasks the scenario mints: the runner's run-wide budget reserves
    # ``per_task_usd x root_tasks`` per attempt (the runtime fences each root task TREE at
    # OUROBOROS_PER_TASK_COST_USD, so SW1's scouts spend under their one root's ceiling).
    root_tasks: int = 1
    # True only for a scenario that LANDS a commit the post-task evolution must absorb and the re-exec
    # restart must serve (SM1): under ``--self-mod`` the runner waits for that absorb and checks it. A
    # scenario that commits nothing (SW1, SK1) has no absorb to wait for or to confirm.
    expects_absorb: bool = False

    def overrides(self, model: str) -> dict:
        out = dict(self.settings_overrides)
        if self.id == "SW1":
            out["OUROBOROS_SUBAGENTS"] = sw1_roster(model)
        if not self.expects_absorb:
            # A lane that commits nothing must not promote either: under --self-mod its one-shot cycle could
            # commit and re-exec the server in the middle of the lifecycle under test (SK1 review/grants/
            # dispatch), turning an unrelated restart into the lane's verdict. Only SM1 exercises evolution.
            out["OUROBOROS_POST_TASK_EVOLUTION"] = "false"
        return out


SCENARIOS: dict[str, Scenario] = {
    "SM1": Scenario(
        "SM1", "Brand accent change lands as a reviewed release through commit_reviewed (advanced, blocking)",
        sm1_prompt(), {"OUROBOROS_RUNTIME_MODE": "advanced", "OUROBOROS_REVIEW_ENFORCEMENT": "blocking"},
        True, run_sm1, sm1_stub_script, expects_absorb=True),
    "SW1": Scenario(
        "SW1", "Swarm: force_plan + roster, two children, fanout receipt, cost rollup, no orphans",
        SW1_OBJECTIVE, {"OUROBOROS_MAX_WORKERS": 4, "OUROBOROS_MAX_SUBAGENT_DEPTH": 1},
        True, run_sw1, sw1_stub_script),
    "SK1": Scenario(
        "SK1", "Skill lifecycle: model authors SKILL.md+plugin.py, preflight, review, grants, enable, dispatch",
        sk1_prompt(), {}, False, run_sk1, sk1_stub_script, root_tasks=2),
}


def diff_sha256(clone: pathlib.Path, pre_head: str, post_head: str) -> str:
    if not pre_head or not post_head or pre_head == post_head:
        return ""
    diff = subprocess.run(["git", "diff", "--binary", pre_head, post_head], cwd=str(clone),
                          check=False, capture_output=True).stdout
    return hashlib.sha256(diff).hexdigest()


def head_sha(clone: pathlib.Path) -> str:
    return _git(["rev-parse", "HEAD"], clone)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
