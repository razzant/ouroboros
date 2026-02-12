# ============================
# Ouroboros — Colab Launcher Cell (pull existing repo + run)
# Fixes: apply_patch shim + no "Drive already mounted" spam
#
# This file is a reference copy of the immutable Colab boot cell.
# The actual boot cell lives in the Colab notebook and must not be
# modified by the agent.  Keep this file in sync manually.
# ============================

import os, sys, json, time, uuid, pathlib, subprocess, datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

# ----------------------------
# 0) Install deps
# ----------------------------
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "openai>=1.0.0", "requests"], check=True)

def ensure_claude_code_cli() -> bool:
    """Best-effort install of Claude Code CLI for Anthropic-powered code edits."""
    local_bin = str(pathlib.Path.home() / ".local" / "bin")
    if local_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{local_bin}:{os.environ.get('PATH', '')}"

    has_cli = subprocess.run(["bash", "-lc", "command -v claude >/dev/null 2>&1"], check=False).returncode == 0
    if has_cli:
        return True

    # Preferred install method (native binary installer).
    subprocess.run(["bash", "-lc", "curl -fsSL https://claude.ai/install.sh | bash"], check=False)
    has_cli = subprocess.run(["bash", "-lc", "command -v claude >/dev/null 2>&1"], check=False).returncode == 0
    if has_cli:
        return True

    # Fallback path for environments where native installer is unavailable.
    subprocess.run(["bash", "-lc", "command -v npm >/dev/null 2>&1 && npm install -g @anthropic-ai/claude-code"], check=False)
    has_cli = subprocess.run(["bash", "-lc", "command -v claude >/dev/null 2>&1"], check=False).returncode == 0
    return has_cli

# ----------------------------
# 0.1) provide apply_patch shim (so LLM "apply_patch<<PATCH" won't crash)
# ----------------------------
APPLY_PATCH_PATH = pathlib.Path("/usr/local/bin/apply_patch")
APPLY_PATCH_CODE = r"""#!/usr/bin/env python3
import sys
import pathlib

def _norm_line(l: str) -> str:
    # accept both " context" and "context" as context lines
    if l.startswith(" "):
        return l[1:]
    return l

def _find_subseq(hay, needle):
    if not needle:
        return 0
    n = len(needle)
    for i in range(0, len(hay) - n + 1):
        ok = True
        for j in range(n):
            if hay[i + j] != needle[j]:
                ok = False
                break
        if ok:
            return i
    return -1

def _find_subseq_rstrip(hay, needle):
    if not needle:
        return 0
    hay2 = [x.rstrip() for x in hay]
    needle2 = [x.rstrip() for x in needle]
    return _find_subseq(hay2, needle2)

def apply_update_file(path: str, hunks: list[list[str]]):
    p = pathlib.Path(path)
    if not p.exists():
        sys.stderr.write(f"apply_patch: file not found: {path}\n")
        sys.exit(2)

    text = p.read_text(encoding="utf-8")
    src = text.splitlines()

    for hunk in hunks:
        old_seq = []
        new_seq = []
        for line in hunk:
            if line.startswith("+"):
                new_seq.append(line[1:])
            elif line.startswith("-"):
                old_seq.append(line[1:])
            else:
                c = _norm_line(line)
                old_seq.append(c)
                new_seq.append(c)

        idx = _find_subseq(src, old_seq)
        if idx < 0:
            idx = _find_subseq_rstrip(src, old_seq)
        if idx < 0:
            sys.stderr.write("apply_patch: failed to match hunk in file: " + path + "\n")
            sys.stderr.write("HUNK (old_seq):\n" + "\n".join(old_seq) + "\n")
            sys.exit(3)

        src = src[:idx] + new_seq + src[idx + len(old_seq):]

    p.write_text("\n".join(src) + "\n", encoding="utf-8")

def main():
    lines = sys.stdin.read().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]

        if line.startswith("*** Begin Patch"):
            i += 1
            continue

        if line.startswith("*** Update File:"):
            path = line.split(":", 1)[1].strip()
            i += 1

            hunks = []
            cur = []
            while i < len(lines) and not lines[i].startswith("*** "):
                if lines[i].startswith("@@"):
                    if cur:
                        hunks.append(cur)
                        cur = []
                    i += 1
                    continue
                cur.append(lines[i])
                i += 1
            if cur:
                hunks.append(cur)

            apply_update_file(path, hunks)
            continue

        if line.startswith("*** End Patch"):
            i += 1
            continue

        # ignore unknown lines/blocks
        i += 1

if __name__ == "__main__":
    main()
"""
APPLY_PATCH_PATH.parent.mkdir(parents=True, exist_ok=True)
APPLY_PATCH_PATH.write_text(APPLY_PATCH_CODE, encoding="utf-8")
APPLY_PATCH_PATH.chmod(0o755)

# ----------------------------
# 1) Secrets (Colab userdata -> env fallback)
# ----------------------------
from google.colab import userdata  # type: ignore
from google.colab import drive  # type: ignore

def get_secret(name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    v = userdata.get(name)
    if v is None:
        v = os.environ.get(name, default)
    if required:
        assert v is not None and str(v).strip() != "", f"Missing required secret: {name}"
    return v

OPENROUTER_API_KEY = get_secret("OPENROUTER_API_KEY", required=True)
TELEGRAM_BOT_TOKEN = get_secret("TELEGRAM_BOT_TOKEN", required=True)
TOTAL_BUDGET_DEFAULT = get_secret("TOTAL_BUDGET", required=True)
GITHUB_TOKEN = get_secret("GITHUB_TOKEN", required=True)

OPENAI_API_KEY = get_secret("OPENAI_API_KEY", default="")  # optional
ANTHROPIC_API_KEY = get_secret("ANTHROPIC_API_KEY", default="")  # optional; enables Claude Code CLI tool

GITHUB_USER = get_secret("GITHUB_USER", default="razzant")
GITHUB_REPO = get_secret("GITHUB_REPO", default="ouroboros")

MAX_WORKERS = int(get_secret("OUROBOROS_MAX_WORKERS", default="5") or "5")
MODEL_MAIN = get_secret("OUROBOROS_MODEL", default="openai/gpt-5.2")

def as_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off", ""):
        return False
    return default

IDLE_ENABLED = as_bool(get_secret("OUROBOROS_IDLE_ENABLED", default="1"), default=True)
IDLE_COOLDOWN_SEC = max(60, int(get_secret("OUROBOROS_IDLE_COOLDOWN_SEC", default="900") or "900"))
IDLE_BUDGET_PCT_CAP = max(1.0, min(float(get_secret("OUROBOROS_IDLE_BUDGET_PCT_CAP", default="35") or "35"), 100.0))
IDLE_MAX_PER_DAY = max(1, int(get_secret("OUROBOROS_IDLE_MAX_PER_DAY", default="8") or "8"))
EVOLUTION_ENABLED_BY_DEFAULT = as_bool(get_secret("OUROBOROS_EVOLUTION_ENABLED_BY_DEFAULT", default="0"), default=False)
BUDGET_REPORT_EVERY_MESSAGES = max(1, int(get_secret("OUROBOROS_BUDGET_REPORT_EVERY_MESSAGES", default="10") or "10"))

# expose needed env to workers (do not print)
os.environ["OPENROUTER_API_KEY"] = str(OPENROUTER_API_KEY)
os.environ["OPENAI_API_KEY"] = str(OPENAI_API_KEY or "")
os.environ["ANTHROPIC_API_KEY"] = str(ANTHROPIC_API_KEY or "")
os.environ["OUROBOROS_MODEL"] = str(MODEL_MAIN or "openai/gpt-5.2")
os.environ["TELEGRAM_BOT_TOKEN"] = str(TELEGRAM_BOT_TOKEN)  # to support agent-side UX like typing indicator

# Install Claude Code CLI only when Anthropic API access is configured.
if str(ANTHROPIC_API_KEY or "").strip():
    ensure_claude_code_cli()

# ----------------------------
# 2) Mount Drive (quietly)
# ----------------------------
if not pathlib.Path("/content/drive/MyDrive").exists():
    drive.mount("/content/drive")

DRIVE_ROOT = pathlib.Path("/content/drive/MyDrive/Ouroboros").resolve()
REPO_DIR = pathlib.Path("/content/ouroboros_repo").resolve()

for sub in ["state", "logs", "memory", "index", "locks", "archive"]:
    (DRIVE_ROOT / sub).mkdir(parents=True, exist_ok=True)
REPO_DIR.mkdir(parents=True, exist_ok=True)

STATE_PATH = DRIVE_ROOT / "state" / "state.json"

def ensure_state_defaults(st: Dict[str, Any]) -> Dict[str, Any]:
    st.setdefault("created_at", datetime.datetime.now(datetime.timezone.utc).isoformat())
    st.setdefault("owner_id", None)
    st.setdefault("owner_chat_id", None)
    st.setdefault("tg_offset", 0)
    st.setdefault("spent_usd", 0.0)
    st.setdefault("spent_calls", 0)
    st.setdefault("spent_tokens_prompt", 0)
    st.setdefault("spent_tokens_completion", 0)
    st.setdefault("approvals", {})
    st.setdefault("session_id", uuid.uuid4().hex)
    st.setdefault("current_branch", None)
    st.setdefault("current_sha", None)
    st.setdefault("last_owner_message_at", "")
    st.setdefault("last_idle_task_at", "")
    st.setdefault("last_evolution_task_at", "")
    st.setdefault("idle_cursor", 0)
    st.setdefault("budget_messages_since_report", 0)
    st.setdefault("evolution_mode_enabled", EVOLUTION_ENABLED_BY_DEFAULT)
    st.setdefault("evolution_cycle", 0)
    if not isinstance(st.get("idle_stats"), dict):
        st["idle_stats"] = {}
    return st

def load_state() -> Dict[str, Any]:
    if STATE_PATH.exists():
        st = ensure_state_defaults(json.loads(STATE_PATH.read_text(encoding="utf-8")))
        return st
    st = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "owner_id": None,
        "owner_chat_id": None,
        "tg_offset": 0,
        "spent_usd": 0.0,
        "spent_calls": 0,
        "spent_tokens_prompt": 0,
        "spent_tokens_completion": 0,
        "approvals": {},
        "session_id": uuid.uuid4().hex,
        "current_branch": None,
        "current_sha": None,
        "last_owner_message_at": "",
        "last_idle_task_at": "",
        "last_evolution_task_at": "",
        "idle_cursor": 0,
        "budget_messages_since_report": 0,
        "evolution_mode_enabled": EVOLUTION_ENABLED_BY_DEFAULT,
        "evolution_cycle": 0,
        "idle_stats": {},
    }
    STATE_PATH.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
    return st

def save_state(st: Dict[str, Any]) -> None:
    st = ensure_state_defaults(st)
    STATE_PATH.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")

def append_jsonl(path: pathlib.Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

CHAT_LOG_PATH = DRIVE_ROOT / "logs" / "chat.jsonl"
if not CHAT_LOG_PATH.exists():
    CHAT_LOG_PATH.write_text("", encoding="utf-8")

# ----------------------------
# 3) Git: clone/pull repo (no creation), dev->stable fallback
# ----------------------------
BRANCH_DEV = "ouroboros"
BRANCH_STABLE = "ouroboros-stable"

REMOTE_URL = f"https://{GITHUB_TOKEN}:x-oauth-basic@github.com/{GITHUB_USER}/{GITHUB_REPO}.git"

def ensure_repo_present() -> None:
    if not (REPO_DIR / ".git").exists():
        subprocess.run(["rm", "-rf", str(REPO_DIR)], check=False)
        subprocess.run(["git", "clone", REMOTE_URL, str(REPO_DIR)], check=True)
    else:
        subprocess.run(["git", "remote", "set-url", "origin", REMOTE_URL], cwd=str(REPO_DIR), check=True)
    subprocess.run(["git", "config", "user.name", "Ouroboros"], cwd=str(REPO_DIR), check=True)
    subprocess.run(["git", "config", "user.email", "ouroboros@users.noreply.github.com"], cwd=str(REPO_DIR), check=True)
    subprocess.run(["git", "fetch", "origin"], cwd=str(REPO_DIR), check=True)

def checkout_and_reset(branch: str) -> None:
    subprocess.run(["git", "checkout", branch], cwd=str(REPO_DIR), check=True)
    subprocess.run(["git", "reset", "--hard", f"origin/{branch}"], cwd=str(REPO_DIR), check=True)
    st = load_state()
    st["current_branch"] = branch
    st["current_sha"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_DIR), capture_output=True, text=True, check=True).stdout.strip()
    save_state(st)

def import_test() -> Dict[str, Any]:
    r = subprocess.run(
        ["python3", "-c", "import ouroboros, ouroboros.agent; print('import_ok')"],
        cwd=str(REPO_DIR),
        capture_output=True,
        text=True,
    )
    return {"ok": (r.returncode == 0), "stdout": r.stdout, "stderr": r.stderr, "returncode": r.returncode}

ensure_repo_present()
checkout_and_reset(BRANCH_DEV)
t = import_test()
if not t["ok"]:
    append_jsonl(DRIVE_ROOT / "logs" / "supervisor.jsonl", {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "type": "import_fail_dev",
        "stdout": t["stdout"],
        "stderr": t["stderr"],
    })
    checkout_and_reset(BRANCH_STABLE)
    t2 = import_test()
    assert t2["ok"], f"Stable branch also failed import.\n\nSTDOUT:\n{t2['stdout']}\n\nSTDERR:\n{t2['stderr']}"

# ----------------------------
# 4) Telegram (long polling)
# ----------------------------
class TelegramClient:
    def __init__(self, token: str):
        self.base = f"https://api.telegram.org/bot{token}"

    def get_updates(self, offset: int, timeout: int = 10) -> List[Dict[str, Any]]:
        last_err = "unknown"
        for attempt in range(3):
            try:
                r = requests.get(
                    f"{self.base}/getUpdates",
                    params={"offset": offset, "timeout": timeout, "allowed_updates": ["message", "edited_message"]},
                    timeout=timeout + 5,
                )
                r.raise_for_status()
                data = r.json()
                if data.get("ok") is not True:
                    raise RuntimeError(f"Telegram getUpdates failed: {data}")
                return data.get("result") or []
            except Exception as e:
                last_err = repr(e)
                if attempt < 2:
                    time.sleep(0.8 * (attempt + 1))
        raise RuntimeError(f"Telegram getUpdates failed after retries: {last_err}")

    def send_message(self, chat_id: int, text: str) -> Tuple[bool, str]:
        last_err = "unknown"
        for attempt in range(3):
            try:
                r = requests.post(
                    f"{self.base}/sendMessage",
                    data={"chat_id": chat_id, "text": text, "disable_web_page_preview": True},
                    timeout=30,
                )
                r.raise_for_status()
                data = r.json()
                if data.get("ok") is True:
                    return True, "ok"
                last_err = f"telegram_api_error: {data}"
            except Exception as e:
                last_err = repr(e)

            if attempt < 2:
                time.sleep(0.8 * (attempt + 1))

        return False, last_err

TG = TelegramClient(str(TELEGRAM_BOT_TOKEN))

def split_telegram(text: str, limit: int = 3800) -> List[str]:
    chunks: List[str] = []
    s = text
    while len(s) > limit:
        cut = s.rfind("\n", 0, limit)
        if cut < 100:
            cut = limit
        chunks.append(s[:cut])
        s = s[cut:]
    chunks.append(s)
    return chunks

def _format_budget_line(st: Dict[str, Any]) -> str:
    spent = float(st.get("spent_usd") or 0.0)
    total = float(get_secret("TOTAL_BUDGET", default=str(TOTAL_BUDGET_DEFAULT)) or TOTAL_BUDGET_DEFAULT)
    pct = (spent / total * 100.0) if total > 0 else 0.0
    sha = (st.get("current_sha") or "")[:8]
    branch = st.get("current_branch") or "?"
    return f"—\nBudget: ${spent:.4f} / ${total:.2f} ({pct:.2f}%) | {branch}@{sha}"


def budget_line(force: bool = False) -> str:
    """Return budget line every N outgoing messages.

    - force=True always prints and resets the message counter.
    - default cadence comes from OUROBOROS_BUDGET_REPORT_EVERY_MESSAGES (default: 10).
    """
    try:
        st = load_state()
        every = max(1, int(BUDGET_REPORT_EVERY_MESSAGES))
        if force:
            st["budget_messages_since_report"] = 0
            save_state(st)
            return _format_budget_line(st)

        counter = int(st.get("budget_messages_since_report") or 0) + 1
        if counter < every:
            st["budget_messages_since_report"] = counter
            save_state(st)
            return ""

        st["budget_messages_since_report"] = 0
        save_state(st)
        return _format_budget_line(st)
    except Exception:
        # Never fail message sending because of budget reporting.
        return ""

def log_chat(direction: str, chat_id: int, user_id: int, text: str) -> None:
    append_jsonl(DRIVE_ROOT / "logs" / "chat.jsonl", {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "session_id": load_state().get("session_id"),
        "direction": direction,
        "chat_id": chat_id,
        "user_id": user_id,
        "text": text,
    })

def send_with_budget(chat_id: int, text: str, log_text: Optional[str] = None, force_budget: bool = False) -> None:
    st = load_state()
    owner_id = int(st.get("owner_id") or 0)
    log_chat("out", chat_id, owner_id, text if log_text is None else log_text)
    budget = budget_line(force=force_budget)
    _text = str(text or "")
    # If we already sent the main message directly from the worker, it may pass a zero-width space (\u200b)
    # to ask the supervisor to send only the budget line. If budget is not due, skip sending to avoid blank messages.
    if not budget:
        if _text.strip() in ("", "\u200b"):
            return
        full = _text
    else:
        base = _text.rstrip()
        if base in ("", "\u200b"):
            full = budget
        else:
            full = base + "\n\n" + budget
    for idx, part in enumerate(split_telegram(full)):
        ok, err = TG.send_message(chat_id, part)
        if not ok:
            append_jsonl(
                DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {
                    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "type": "telegram_send_error",
                    "chat_id": chat_id,
                    "part_index": idx,
                    "error": err,
                },
            )
            break

# ----------------------------
# 4.5) Router: direct answer vs full agent task
# ----------------------------
ROUTER_SYSTEM_PROMPT = """You are Ouroboros (Уроборос), a self-modifying AI agent.
Your job RIGHT NOW: decide whether to answer the user directly or route to the full agent pipeline.

ANSWER DIRECTLY (just write the answer) if:
- Simple questions (math, facts, definitions, opinions)
- Casual conversation, greetings, thanks
- General knowledge questions
- Explaining concepts
- Questions about yourself ONLY when they are generic and don't require checking runtime state

RESPOND WITH EXACTLY "NEEDS_TASK" on the FIRST LINE if the message requires:
- Reading or writing files, code, configs
- Git operations (commit, push, diff, status)
- Web search for fresh/current information
- Log analysis, examining Drive files
- Code changes or self-modification
- Running shell commands
- Any tool or system access
- Analyzing repository contents
- Checking current runtime state/capabilities (available tools, CLI presence, current branch/version, recent action results)
- Anything you're unsure about

When answering directly, respond in the user's language. Be concise and helpful.
When routing to task, write NEEDS_TASK on the first line, then optionally a brief reason."""

def route_and_maybe_answer(text: str) -> Optional[str]:
    """Quick LLM call: return direct answer or None (meaning 'create a full task')."""
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            default_headers={"HTTP-Referer": "https://colab.research.google.com/", "X-Title": "Ouroboros-Router"},
        )

        # Minimal context: last ~10 chat messages for conversational continuity
        recent_chat = ""
        if CHAT_LOG_PATH.exists():
            try:
                lines = CHAT_LOG_PATH.read_text(encoding="utf-8").strip().split("\n")
                recent_lines = lines[-10:] if len(lines) > 10 else lines
                recent_chat = "\n".join(recent_lines)
            except Exception:
                pass

        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        ]
        if recent_chat:
            messages.append({"role": "system", "content": f"Recent chat context (JSONL):\n{recent_chat}"})
        messages.append({"role": "user", "content": text})

        resp = client.chat.completions.create(
            model=os.environ.get("OUROBOROS_MODEL", "openai/gpt-5.2"),
            messages=messages,
            max_tokens=2000,
        )

        # Track router cost
        usage = (resp.model_dump().get("usage") or {})
        update_budget_from_usage(usage)

        answer = (resp.choices[0].message.content or "").strip()
        if answer.startswith("NEEDS_TASK"):
            return None
        return answer
    except Exception as e:
        # On any error, fall through to task creation
        append_jsonl(DRIVE_ROOT / "logs" / "events.jsonl", {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "type": "router_error", "error": repr(e),
        })
        return None

# ----------------------------
# 5) Workers + strict FIFO queue
# ----------------------------
import multiprocessing as mp
CTX = mp.get_context("fork")

@dataclass
class Worker:
    wid: int
    proc: mp.Process
    in_q: Any
    busy_task_id: Optional[str] = None

EVENT_Q = CTX.Queue()
WORKERS: Dict[int, Worker] = {}
PENDING: List[Dict[str, Any]] = []
RUNNING: Dict[str, Dict[str, Any]] = {}
CRASH_TS: List[float] = []

def worker_main(wid: int, in_q: Any, out_q: Any, repo_dir: str, drive_root: str) -> None:
    import sys as _sys
    _sys.path.insert(0, repo_dir)
    from ouroboros.agent import make_agent  # type: ignore
    agent = make_agent(repo_dir=repo_dir, drive_root=drive_root, event_queue=out_q)
    while True:
        task = in_q.get()
        if task is None or task.get("type") == "shutdown":
            break
        events = agent.handle_task(task)
        for e in events:
            e2 = dict(e)
            e2["worker_id"] = wid
            out_q.put(e2)

def spawn_workers(n: int) -> None:
    WORKERS.clear()
    for i in range(n):
        in_q = CTX.Queue()
        proc = CTX.Process(target=worker_main, args=(i, in_q, EVENT_Q, str(REPO_DIR), str(DRIVE_ROOT)))
        proc.daemon = True
        proc.start()
        WORKERS[i] = Worker(wid=i, proc=proc, in_q=in_q, busy_task_id=None)

def kill_workers() -> None:
    for w in WORKERS.values():
        if w.proc.is_alive():
            w.proc.terminate()
    for w in WORKERS.values():
        w.proc.join(timeout=5)
    WORKERS.clear()

def assign_tasks() -> None:
    for w in WORKERS.values():
        if w.busy_task_id is None and PENDING:
            task = PENDING.pop(0)
            w.busy_task_id = task["id"]
            w.in_q.put(task)
            RUNNING[task["id"]] = task
            st = load_state()
            if st.get("owner_chat_id"):
                send_with_budget(int(st["owner_chat_id"]), f"▶️ Стартую задачу {task['id']} (worker {w.wid})")

def update_budget_from_usage(usage: Dict[str, Any]) -> None:
    def _to_float(v: Any, default: float = 0.0) -> float:
        try:
            return float(v)
        except Exception:
            return default

    def _to_int(v: Any, default: int = 0) -> int:
        try:
            return int(v)
        except Exception:
            return default

    st = load_state()
    cost = usage.get("cost") if isinstance(usage, dict) else None
    if cost is None:
        cost = 0.0
    st["spent_usd"] = _to_float(st.get("spent_usd") or 0.0) + _to_float(cost)
    st["spent_calls"] = int(st.get("spent_calls") or 0) + 1
    st["spent_tokens_prompt"] = _to_int(st.get("spent_tokens_prompt") or 0) + _to_int(usage.get("prompt_tokens") if isinstance(usage, dict) else 0)
    st["spent_tokens_completion"] = _to_int(st.get("spent_tokens_completion") or 0) + _to_int(usage.get("completion_tokens") if isinstance(usage, dict) else 0)
    save_state(st)

def parse_iso_to_ts(iso_ts: str) -> Optional[float]:
    txt = str(iso_ts or "").strip()
    if not txt:
        return None
    try:
        return datetime.datetime.fromisoformat(txt.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None

def budget_pct(st: Dict[str, Any]) -> float:
    spent = float(st.get("spent_usd") or 0.0)
    total = float(get_secret("TOTAL_BUDGET", default=str(TOTAL_BUDGET_DEFAULT)) or TOTAL_BUDGET_DEFAULT)
    if total <= 0:
        return 0.0
    return (spent / total) * 100.0

def _load_evolution_prompt_text() -> str:
    p = REPO_DIR / "prompts" / "evolution.md"
    try:
        txt = p.read_text(encoding="utf-8").strip()
        if txt:
            return txt
    except Exception:
        pass
    return (
        "Endless evolution mode is active.\n"
        "- Do one high-impact self-improvement step per cycle.\n"
        "- Use LLM-based reasoning over hardcoded response templates.\n"
        "- Commit+push only to branch ouroboros, then request_restart.\n"
        "- Report: done/result/next.\n"
    )

def build_evolution_task_text(cycle: int) -> str:
    return (
        f"ENDLESS EVOLUTION CYCLE #{cycle}\n\n"
        "Mode is active until owner asks to stop.\n"
        "Start with `repo_read('prompts/evolution.md')` and follow it exactly.\n"
        "Do one high-leverage self-improvement step now.\n"
        "Strict branch rule: only `ouroboros` for any write/commit/push; never touch `main` or `ouroboros-stable`.\n"
        "After changes: verify, commit+push, request_restart, then report concise Done/Result/Next.\n\n"
        "Prompt snapshot:\n"
        + _load_evolution_prompt_text()
    )

def enqueue_evolution_task_if_needed() -> None:
    if PENDING or RUNNING:
        return

    st = load_state()
    if not bool(st.get("evolution_mode_enabled")):
        return

    owner_chat_id = st.get("owner_chat_id")
    if not owner_chat_id:
        return

    if budget_pct(st) >= 100.0:
        st["evolution_mode_enabled"] = False
        save_state(st)
        append_jsonl(
            DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "type": "evolution_mode_auto_stopped",
                "reason": "budget_exhausted",
                "budget_pct": budget_pct(st),
            },
        )
        send_with_budget(int(owner_chat_id), "💸 Endless evolution остановлен: бюджет достиг лимита.")
        return

    cycle = int(st.get("evolution_cycle") or 0) + 1
    tid = uuid.uuid4().hex[:8]
    PENDING.append(
        {
            "id": tid,
            "type": "evolution",
            "chat_id": int(owner_chat_id),
            "text": build_evolution_task_text(cycle),
        }
    )

    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    st["evolution_cycle"] = cycle
    st["last_evolution_task_at"] = now_iso
    save_state(st)

    append_jsonl(
        DRIVE_ROOT / "logs" / "supervisor.jsonl",
        {
            "ts": now_iso,
            "type": "evolution_task_enqueued",
            "task_id": tid,
            "cycle": cycle,
            "budget_pct": budget_pct(st),
        },
    )
    send_with_budget(int(owner_chat_id), f"🧬 Evolution task queued: {tid} (cycle {cycle})")

def idle_task_catalog() -> List[Tuple[str, str]]:
    return [
        (
            "memory_consolidation",
            "Idle internal task: consolidate working memory. Update memory/scratchpad.md from recent logs and add compact evidence quotes.",
        ),
        (
            "performance_analysis",
            "Idle internal task: analyze recent tools/events logs and report key bottlenecks, recurring failures, and optimization opportunities.",
        ),
        (
            "code_improvement_idea",
            "Idle internal task: inspect your own codebase and propose one high-impact improvement with rationale and validation plan.",
        ),
        (
            "web_learning",
            "Idle internal task: use web_search for one focused topic that can improve reliability/efficiency of this system, then summarize practical takeaways.",
        ),
        (
            "owner_idea_proposal",
            "Idle internal task: prepare one concise proactive idea for the owner based on current priorities and unresolved threads.",
        ),
    ]

def enqueue_idle_task_if_needed() -> None:
    if not IDLE_ENABLED:
        return
    if PENDING or RUNNING:
        return

    st = load_state()
    if bool(st.get("evolution_mode_enabled")):
        return
    owner_chat_id = st.get("owner_chat_id")
    if not owner_chat_id:
        return

    now = time.time()
    last_owner_ts = parse_iso_to_ts(str(st.get("last_owner_message_at") or ""))
    if last_owner_ts is not None and (now - last_owner_ts) < IDLE_COOLDOWN_SEC:
        return

    last_idle_ts = parse_iso_to_ts(str(st.get("last_idle_task_at") or ""))
    if last_idle_ts is not None and (now - last_idle_ts) < IDLE_COOLDOWN_SEC:
        return

    if budget_pct(st) >= IDLE_BUDGET_PCT_CAP:
        return

    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    idle_stats = st.get("idle_stats") if isinstance(st.get("idle_stats"), dict) else {}
    day_stat = idle_stats.get(today) if isinstance(idle_stats.get(today), dict) else {}
    day_count = int(day_stat.get("count") or 0)
    if day_count >= IDLE_MAX_PER_DAY:
        return

    catalog = idle_task_catalog()
    cursor = int(st.get("idle_cursor") or 0)
    kind, text = catalog[cursor % len(catalog)]
    tid = uuid.uuid4().hex[:8]
    PENDING.append({"id": tid, "type": "idle", "chat_id": int(owner_chat_id), "text": text})

    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    st["idle_cursor"] = cursor + 1
    st["last_idle_task_at"] = now_iso
    idle_stats[today] = {
        "count": day_count + 1,
        "last_task_id": tid,
        "last_kind": kind,
        "last_at": now_iso,
    }
    # Keep recent days only.
    if len(idle_stats) > 14:
        for d in sorted(idle_stats.keys())[:-14]:
            idle_stats.pop(d, None)
    st["idle_stats"] = idle_stats
    save_state(st)

    append_jsonl(
        DRIVE_ROOT / "logs" / "supervisor.jsonl",
        {
            "ts": now_iso,
            "type": "idle_task_enqueued",
            "task_id": tid,
            "kind": kind,
            "budget_pct": budget_pct(st),
        },
    )
    send_with_budget(int(owner_chat_id), f"🧠 Idle task queued: {tid} ({kind})")

def respawn_worker(wid: int) -> None:
    in_q = CTX.Queue()
    proc = CTX.Process(target=worker_main, args=(wid, in_q, EVENT_Q, str(REPO_DIR), str(DRIVE_ROOT)))
    proc.daemon = True
    proc.start()
    WORKERS[wid] = Worker(wid=wid, proc=proc, in_q=in_q, busy_task_id=None)

def ensure_workers_healthy() -> None:
    for wid, w in list(WORKERS.items()):
        if not w.proc.is_alive():
            CRASH_TS.append(time.time())
            if w.busy_task_id and w.busy_task_id in RUNNING:
                task = RUNNING.pop(w.busy_task_id)
                PENDING.insert(0, task)
            respawn_worker(wid)

    now = time.time()
    CRASH_TS[:] = [t for t in CRASH_TS if (now - t) < 60.0]
    # if crash storm, fallback to stable branch (import must work)
    if len(CRASH_TS) >= 3:
        st = load_state()
        if st.get("owner_chat_id"):
            send_with_budget(int(st["owner_chat_id"]), "⚠️ Частые падения воркеров. Переключаюсь на ouroboros-stable и перезапускаюсь.")
        checkout_and_reset(BRANCH_STABLE)
        kill_workers()
        spawn_workers(MAX_WORKERS)
        CRASH_TS.clear()

def rotate_chat_log_if_needed(max_bytes: int = 800_000) -> None:
    chat = DRIVE_ROOT / "logs" / "chat.jsonl"
    if not chat.exists():
        return
    if chat.stat().st_size < max_bytes:
        return
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_path = DRIVE_ROOT / "archive" / f"chat_{ts}.jsonl"
    archive_path.write_bytes(chat.read_bytes())
    chat.write_text("", encoding="utf-8")

def status_text() -> str:
    st = load_state()
    lines = []
    lines.append(f"owner_id: {st.get('owner_id')}")
    lines.append(f"session_id: {st.get('session_id')}")
    lines.append(f"version: {st.get('current_branch')}@{(st.get('current_sha') or '')[:8]}")
    lines.append(f"workers: {len(WORKERS)} (busy: {sum(1 for w in WORKERS.values() if w.busy_task_id is not None)})")
    lines.append(f"pending: {len(PENDING)}")
    if PENDING:
        lines.append("pending_ids: " + ", ".join([t["id"] for t in PENDING[:10]]))
    busy = [f"{w.wid}:{w.busy_task_id}" for w in WORKERS.values() if w.busy_task_id]
    if busy:
        lines.append("busy: " + ", ".join(busy))
    lines.append(f"spent_usd: {st.get('spent_usd')}")
    lines.append(f"spent_calls: {st.get('spent_calls')}")
    lines.append(f"prompt_tokens: {st.get('spent_tokens_prompt')}, completion_tokens: {st.get('spent_tokens_completion')}")
    lines.append(f"budget_report_every_messages: {BUDGET_REPORT_EVERY_MESSAGES}")
    lines.append(
        "idle: "
        + f"enabled={int(IDLE_ENABLED)}, cooldown_sec={IDLE_COOLDOWN_SEC}, "
        + f"budget_cap_pct={IDLE_BUDGET_PCT_CAP:.1f}, max_per_day={IDLE_MAX_PER_DAY}"
    )
    lines.append(
        "evolution: "
        + f"enabled={int(bool(st.get('evolution_mode_enabled')))}, "
        + f"cycle={int(st.get('evolution_cycle') or 0)}"
    )
    lines.append(f"last_owner_message_at: {st.get('last_owner_message_at') or '-'}")
    lines.append(f"last_idle_task_at: {st.get('last_idle_task_at') or '-'}")
    lines.append(f"last_evolution_task_at: {st.get('last_evolution_task_at') or '-'}")
    return "\n".join(lines)

def cancel_task_by_id(task_id: str) -> bool:
    for i, t in enumerate(list(PENDING)):
        if t["id"] == task_id:
            PENDING.pop(i)
            return True
    for w in WORKERS.values():
        if w.busy_task_id == task_id:
            RUNNING.pop(task_id, None)
            if w.proc.is_alive():
                w.proc.terminate()
            w.proc.join(timeout=5)
            respawn_worker(w.wid)
            return True
    return False

def handle_approval(chat_id: int, text: str) -> bool:
    parts = text.strip().split()
    if not parts:
        return False
    cmd = parts[0].lower()
    if cmd not in ("/approve", "/deny"):
        return False
    assert len(parts) >= 2, "Usage: /approve <approval_id> or /deny <approval_id>"
    approval_id = parts[1].strip()
    st = load_state()
    approvals = st.get("approvals") or {}
    assert approval_id in approvals, f"Unknown approval_id: {approval_id}"
    approvals[approval_id]["status"] = "approved" if cmd == "/approve" else "denied"
    st["approvals"] = approvals
    save_state(st)
    send_with_budget(chat_id, f"OK: {cmd} {approval_id}")

    # Execute approved actions
    if cmd == "/approve" and approvals[approval_id].get("type") == "stable_promotion":
        try:
            subprocess.run(["git", "fetch", "origin"], cwd=str(REPO_DIR), check=True)
            subprocess.run(["git", "push", "origin", f"{BRANCH_DEV}:{BRANCH_STABLE}"], cwd=str(REPO_DIR), check=True)
            new_sha = subprocess.run(["git", "rev-parse", f"origin/{BRANCH_STABLE}"], cwd=str(REPO_DIR), capture_output=True, text=True, check=True).stdout.strip()
            send_with_budget(chat_id, f"✅ Промоут выполнен: {BRANCH_DEV} → {BRANCH_STABLE} ({new_sha[:8]})")
        except Exception as e:
            send_with_budget(chat_id, f"❌ Ошибка промоута в stable: {e}")

    if cmd == "/approve" and approvals[approval_id].get("type") == "reindex":
        reason = str(approvals[approval_id].get("reason") or "").strip()
        tid = uuid.uuid4().hex[:8]
        PENDING.append(
            {
                "id": tid,
                "type": "task",
                "chat_id": chat_id,
                "text": (
                    "Approved internal task: run full reindex of drive/index/summaries.json. "
                    "Rebuild summaries carefully, report what changed, and include validation checks. "
                    f"Reason: {reason}"
                ).strip(),
            }
        )
        send_with_budget(chat_id, f"✅ Reindex approval accepted. Queued task {tid}.")

    return True

# start
kill_workers()
spawn_workers(MAX_WORKERS)

append_jsonl(DRIVE_ROOT / "logs" / "supervisor.jsonl", {
    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "type": "launcher_start",
    "branch": load_state().get("current_branch"),
    "sha": load_state().get("current_sha"),
    "max_workers": MAX_WORKERS,
    "idle_enabled": IDLE_ENABLED,
    "idle_cooldown_sec": IDLE_COOLDOWN_SEC,
    "idle_budget_pct_cap": IDLE_BUDGET_PCT_CAP,
    "idle_max_per_day": IDLE_MAX_PER_DAY,
    "evolution_enabled_by_default": int(EVOLUTION_ENABLED_BY_DEFAULT),
    "budget_report_every_messages": BUDGET_REPORT_EVERY_MESSAGES,
})

offset = int(load_state().get("tg_offset") or 0)

while True:
    rotate_chat_log_if_needed()
    ensure_workers_healthy()

    # Drain worker events
    while EVENT_Q.qsize() > 0:
        evt = EVENT_Q.get()
        et = evt.get("type")

        if et == "llm_usage":
            update_budget_from_usage(evt.get("usage") or {})
            continue

        if et == "send_message":
            try:
                _log_text = evt.get("log_text")
                send_with_budget(
                    int(evt["chat_id"]),
                    str(evt.get("text") or ""),
                    log_text=(str(_log_text) if isinstance(_log_text, str) else None),
                )
            except Exception as e:
                append_jsonl(
                    DRIVE_ROOT / "logs" / "supervisor.jsonl",
                    {
                        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "type": "send_message_event_error",
                        "error": repr(e),
                    },
                )
            continue

        if et == "task_done":
            task_id = evt.get("task_id")
            wid = evt.get("worker_id")
            if task_id:
                RUNNING.pop(str(task_id), None)
            if wid in WORKERS and WORKERS[wid].busy_task_id == task_id:
                WORKERS[wid].busy_task_id = None
            continue

        if et == "restart_request":
            st = load_state()
            if st.get("owner_chat_id"):
                send_with_budget(int(st["owner_chat_id"]), f"♻️ Restart requested by agent: {evt.get('reason')}")
            checkout_and_reset(BRANCH_DEV)
            it = import_test()
            if not it["ok"]:
                checkout_and_reset(BRANCH_STABLE)
            kill_workers()
            spawn_workers(MAX_WORKERS)
            continue

        if et == "stable_promotion_request":
            approval_id = uuid.uuid4().hex[:8]
            st = load_state()
            approvals = st.get("approvals") or {}
            approvals[approval_id] = {
                "type": "stable_promotion",
                "reason": evt.get("reason", ""),
                "status": "pending",
                "requested_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            st["approvals"] = approvals
            save_state(st)
            if st.get("owner_chat_id"):
                send_with_budget(
                    int(st["owner_chat_id"]),
                    f"🔄 Запрос на промоут в stable:\n{evt.get('reason', '')}\n\n"
                    f"Подтвердить: /approve {approval_id}\n"
                    f"Отклонить: /deny {approval_id}"
                )
            continue

        if et == "schedule_task":
            st = load_state()
            owner_chat_id = st.get("owner_chat_id")
            desc = str(evt.get("description") or "").strip()
            if owner_chat_id and desc:
                tid = uuid.uuid4().hex[:8]
                PENDING.append(
                    {
                        "id": tid,
                        "type": "task",
                        "chat_id": int(owner_chat_id),
                        "text": desc,
                    }
                )
                send_with_budget(int(owner_chat_id), f"🗓️ Запланировал задачу {tid}: {desc}")
            append_jsonl(
                DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {
                    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "type": "schedule_task_event",
                    "description": desc,
                },
            )
            continue

        if et == "cancel_task":
            task_id = str(evt.get("task_id") or "").strip()
            st = load_state()
            owner_chat_id = st.get("owner_chat_id")
            ok = cancel_task_by_id(task_id) if task_id else False
            if owner_chat_id:
                send_with_budget(int(owner_chat_id), f"{'✅' if ok else '❌'} cancel {task_id or '?'} (event)")
            append_jsonl(
                DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {
                    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "type": "cancel_task_event",
                    "task_id": task_id,
                    "ok": ok,
                },
            )
            continue

        if et == "reindex_request":
            approval_id = uuid.uuid4().hex[:8]
            st = load_state()
            approvals = st.get("approvals") or {}
            approvals[approval_id] = {
                "type": "reindex",
                "reason": evt.get("reason", ""),
                "status": "pending",
                "requested_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            st["approvals"] = approvals
            save_state(st)
            if st.get("owner_chat_id"):
                send_with_budget(
                    int(st["owner_chat_id"]),
                    f"🗂️ Запрос на полную реиндексацию:\n{evt.get('reason', '')}\n\n"
                    f"Подтвердить: /approve {approval_id}\n"
                    f"Отклонить: /deny {approval_id}",
                )
            continue

    enqueue_evolution_task_if_needed()
    enqueue_idle_task_if_needed()
    assign_tasks()

    # Poll Telegram
    try:
        updates = TG.get_updates(offset=offset, timeout=10)
    except Exception as e:
        append_jsonl(
            DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "type": "telegram_poll_error",
                "offset": offset,
                "error": repr(e),
            },
        )
        time.sleep(1.5)
        continue
    for upd in updates:
        offset = int(upd["update_id"]) + 1
        msg = upd.get("message") or upd.get("edited_message") or {}
        if not msg:
            continue

        chat_id = int(msg["chat"]["id"])
        from_user = msg.get("from") or {}
        user_id = int(from_user.get("id") or 0)
        text = str(msg.get("text") or "")
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        st = load_state()
        if st.get("owner_id") is None:
            st["owner_id"] = user_id
            st["owner_chat_id"] = chat_id
            st["last_owner_message_at"] = now_iso
            save_state(st)
            log_chat("in", chat_id, user_id, text)
            send_with_budget(chat_id, "✅ Owner registered. Ouroboros online.")
            continue

        if user_id != int(st.get("owner_id")):
            continue

        log_chat("in", chat_id, user_id, text)
        st["last_owner_message_at"] = now_iso
        save_state(st)

        # immutable supervisor commands
        if text.strip().lower().startswith("/panic"):
            send_with_budget(chat_id, "🛑 PANIC: stopping everything now.")
            kill_workers()
            st2 = load_state()
            st2["tg_offset"] = offset
            save_state(st2)
            raise SystemExit("PANIC")

        if text.strip().lower().startswith("/restart"):
            st2 = load_state()
            st2["session_id"] = uuid.uuid4().hex
            save_state(st2)
            send_with_budget(chat_id, "♻️ Restarting (soft).")
            checkout_and_reset(BRANCH_DEV)
            it = import_test()
            if not it["ok"]:
                checkout_and_reset(BRANCH_STABLE)
            kill_workers()
            spawn_workers(MAX_WORKERS)
            continue

        if text.strip().lower().startswith("/status"):
            send_with_budget(chat_id, status_text(), force_budget=True)
            continue

        lowered = text.strip().lower()
        if lowered.startswith("/evolve"):
            parts = lowered.split()
            action = parts[1] if len(parts) > 1 else "on"
            turn_on = action not in ("off", "stop", "0")

            st2 = load_state()
            st2["evolution_mode_enabled"] = bool(turn_on)
            save_state(st2)

            removed_pending = 0
            if not turn_on:
                before = len(PENDING)
                PENDING[:] = [t for t in PENDING if str(t.get("type")) != "evolution"]
                removed_pending = before - len(PENDING)

            append_jsonl(
                DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {
                    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "type": "evolution_mode_toggle",
                    "enabled": bool(turn_on),
                    "removed_pending": removed_pending,
                    "source_text": text,
                },
            )

            if turn_on:
                send_with_budget(
                    chat_id,
                    "🧬 Endless evolution: ON.\n"
                    "Буду крутить self-improvement циклы до твоей остановки или конца бюджета.\n"
                    "Отключить: /evolve stop",
                )
            else:
                send_with_budget(
                    chat_id,
                    f"🛑 Endless evolution: OFF. Снято pending evolution tasks: {removed_pending}.",
                )
            continue

        if lowered in ("останови эволюцию", "остановить эволюцию", "стоп эволюции", "stop evolution"):
            st2 = load_state()
            st2["evolution_mode_enabled"] = False
            save_state(st2)
            before = len(PENDING)
            PENDING[:] = [t for t in PENDING if str(t.get("type")) != "evolution"]
            removed_pending = before - len(PENDING)
            append_jsonl(
                DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {
                    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "type": "evolution_mode_toggle",
                    "enabled": False,
                    "removed_pending": removed_pending,
                    "source_text": text,
                },
            )
            send_with_budget(
                chat_id,
                f"🛑 Endless evolution остановлен. Снято pending evolution tasks: {removed_pending}.",
            )
            continue

        if handle_approval(chat_id, text):
            continue

        if text.strip().lower().startswith("/cancel"):
            parts = text.strip().split()
            assert len(parts) >= 2, "Usage: /cancel <task_id>"
            ok = cancel_task_by_id(parts[1])
            send_with_budget(chat_id, f"{'✅' if ok else '❌'} cancel {parts[1]}")
            continue

        # Route: direct answer or full agent task
        direct = route_and_maybe_answer(text)
        if direct is not None:
            send_with_budget(chat_id, direct)
        else:
            tid = uuid.uuid4().hex[:8]
            PENDING.append({"id": tid, "type": "task", "chat_id": chat_id, "text": text})
            send_with_budget(chat_id, f"🧾 Принято. В очереди: {tid}. (workers={MAX_WORKERS}, pending={len(PENDING)})")

    st = load_state()
    st["tg_offset"] = offset
    save_state(st)

    time.sleep(0.2)
