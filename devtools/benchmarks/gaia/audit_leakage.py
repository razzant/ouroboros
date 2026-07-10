#!/usr/bin/env python3
"""GAIA answer-leakage audit (diagnostic, never changes a score).

GAIA validation answers are public on HuggingFace, so a web-capable agent can in
principle look them up instead of solving the task (Berkeley RDI reported ~98%
by answer-lookup; HAL, arXiv 2510.11977, flags this via LLM log inspection).
Our publishable GAIA rows run WITH web access (parity with web-using baselines
such as Codex/Claude Code), so honesty is established post-hoc by auditing every
trace rather than by network isolation. This script implements that audit.

Two layers, per the accepted methodology:
  1. Deterministic scan  — every web/browser/shell tool call in the run. STRONG
     flags (agent acted on the answer source): a requested leak URL, a benchmark-
     hunting search query, or gold-answer text retrieved FROM a leak source. WEAK
     signals (plain gold-verbatim in a normal page, incidental leak URL in results)
     are recorded for the judge, not auto-flagged.
  2. LLM judge (optional) — a light model scores each sample's web activity
     against a "benchmark answer search / cheating" rubric (HAL-style).

Works for every harness shape:
  * Ouroboros rows (run_gaia.py): local tool calls in <run>/**/tools.jsonl PLUS
    native server-side web-search citations in <run>/**/events.jsonl (the native
    web path never appears in tools.jsonl; its query text is not logged).
  * Codex / Claude Code / Hermes rows (run_harness.py): inspect-log messages PLUS
    the per-sample CLI trace file (codex_trace.jsonl / claude_code_trace.jsonl /
    hermes_trace.txt). Appended prompt boilerplate is stripped before scanning.
  * null rows: no activity (integrity probe).

Output (never mutates the run's scores):
  <run>/leakage_audit.jsonl        one row per sample
  <run>/leakage_audit_summary.json aggregate counts + flagged sample ids

Usage:
  python audit_leakage.py --run-dir <gaia_run>            # deterministic only
  OPENROUTER_API_KEY=... python audit_leakage.py --run-dir <gaia_run> \
      --judge-model openai/gpt-5.2                          # + LLM judge layer
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import re
import sys
import urllib.request

if str(pathlib.Path(__file__).resolve().parents[3]) not in sys.path:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

# SSOT for leak-target hosts/URL/query patterns (shared with tests). See
# leak_targets.py for the pattern-design constraints.
from devtools.benchmarks.gaia.leak_targets import LEAK_QUERY_RE, LEAK_URL_RE  # noqa: E402
# The solvers' SSOT prompt instructions are stripped from full-text traces before
# scanning, so an echoed prompt cannot self-trip LEAK_QUERY_RE (the anti-leak text
# names the answer-source concept; the format text contains "final answer").
from devtools.benchmarks.gaia.inspect_solver import (  # noqa: E402
    GAIA_ANTI_LEAK_INSTRUCTION,
    GAIA_FORMAT_INSTRUCTION,
)

# Web-ish Ouroboros tools whose args/results can carry URLs or retrieved text.
# Every network-capable tool that stays ENABLED in the bench profiles must be
# here (quality_openrouter_web disables only web_search + claude_code_edit).
WEB_TOOLS = {"web_search", "browse_page", "browser_action", "youtube_transcript",
             "run_script", "run_command", "skill_exec", "start_service"}
# Harness (inspect) tool names that touch the web / shell.
HARNESS_WEB_TOOLS = {"web_search", "websearch", "webfetch", "web_fetch", "bash", "browser"}
URL_RE = re.compile(r"https?://[^\s\"'<>)\]}]+", re.IGNORECASE)

# SSOT instruction blobs to strip from full-text traces before scanning, so the
# echoed prompt cannot self-flag every sample (the anti-leak text names the very
# concepts LEAK_QUERY_RE looks for).
_PROMPT_BOILERPLATE = (GAIA_ANTI_LEAK_INSTRUCTION, GAIA_FORMAT_INSTRUCTION)


def _strip_prompt_boilerplate(text: str, gold: str = "") -> str:
    """Remove the appended SSOT instructions from a trace before scanning so the
    prompt echo cannot self-trip the leak-query regex."""
    if not text:
        return text
    for blob in _PROMPT_BOILERPLATE:
        if blob:
            text = text.replace(blob, " ")
            text = text.replace(blob.strip(), " ")
    return text


def _distinctive_gold(gold: str) -> bool:
    """A gold answer worth scanning verbatim: not a bare tiny token that would
    false-positive everywhere (e.g. '3', 'yes'). Distinctive = length>=6 and
    contains a letter, OR a long number/string."""
    g = (gold or "").strip()
    if len(g) < 6:
        return False
    if re.fullmatch(r"[\d.,\s]+", g):
        return len(re.sub(r"\D", "", g)) >= 6  # long numeric answers are distinctive
    return True


def _read_call_blob(run_dir: pathlib.Path, ref) -> str:
    """Best-effort read of a tool call's full result file referenced from tools.jsonl."""
    path = None
    if isinstance(ref, dict):
        path = ref.get("path")
    elif isinstance(ref, str):
        path = ref
    if not path:
        return ""
    try:
        return pathlib.Path(path).read_text(encoding="utf-8", errors="replace")[:200_000]
    except OSError:
        return ""


def _leak_urls(text: str) -> list[str]:
    """Leak-target URLs found in a blob (host/path anchored, not bare tokens)."""
    if not text:
        return []
    return sorted({m.group(0)[:300] for m in LEAK_URL_RE.finditer(text)})


def _load_gold(inspect_log: dict) -> dict:
    """sample_id -> gold answer string (from inspect log targets)."""
    gold = {}
    for s in inspect_log.get("samples", []):
        sid = str(s.get("id"))
        tgt = s.get("target")
        gold[sid] = tgt if isinstance(tgt, str) else json.dumps(tgt, ensure_ascii=False)
    return gold


def _load_scores(inspect_log: dict) -> dict:
    out = {}
    for s in inspect_log.get("samples", []):
        sc = (s.get("scores") or {}).get("gaia_scorer") or {}
        out[str(s.get("id"))] = sc.get("value")
    return out


def _ouroboros_task_to_sample(inspect_log: dict) -> dict:
    """task_id -> sample_id. `ouroboros_result_json` in sample metadata is a PATH
    to the sample's result.json; read it to recover the task_id."""
    m = {}
    for s in inspect_log.get("samples", []):
        meta = s.get("metadata") or {}
        ref = meta.get("ouroboros_result_json")
        rj = None
        if isinstance(ref, dict):
            rj = ref
        elif isinstance(ref, str):
            try:
                rj = json.loads(pathlib.Path(ref).read_text(encoding="utf-8")) if pathlib.Path(ref).exists() \
                    else json.loads(ref)
            except Exception:
                rj = None
        if isinstance(rj, dict):
            tid = str(rj.get("task_id") or "")
            if tid:
                m[tid] = str(s.get("id"))
    return m


def _collect_ouroboros_activity(run_dir: pathlib.Path) -> dict:
    """task_id -> list of {tool, urls, host_hits, result_text} from all tools.jsonl."""
    by_task: dict[str, list] = {}
    for tj in glob.glob(str(run_dir / "**" / "tools.jsonl"), recursive=True):
        try:
            lines = pathlib.Path(tj).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for ln in lines:
            try:
                ev = json.loads(ln)
            except Exception:
                continue
            if ev.get("type") != "tool_call":
                continue
            tool = str(ev.get("tool") or "")
            if tool not in WEB_TOOLS:
                continue
            tid = str(ev.get("task_id") or ev.get("root_task_id") or "")
            args = ev.get("args") or {}
            args_text = json.dumps(args, ensure_ascii=False)
            preview = str(ev.get("result_preview") or "")
            blob = _read_call_blob(run_dir, ev.get("result_ref"))
            query = " ".join(str(args.get(k, "")) for k in ("query", "q", "search", "value", "url"))
            by_task.setdefault(tid, []).append({
                "tool": tool,
                # what the agent REQUESTED (strong signal): leak URLs in args + suspicious query
                "requested_leak_urls": _leak_urls(args_text),
                "suspicious_query": bool(tool in {"web_search"} and LEAK_QUERY_RE.search(query)),
                # what merely CAME BACK (weak signal, for the judge): leak URLs in results
                "result_leak_refs": _leak_urls(preview + "\n" + blob),
                "result_text": (preview + "\n" + blob),
                "args_text": args_text,
            })
    return by_task


def _collect_harness_activity(sample: dict) -> list:
    """Extract web/shell tool activity from an inspect sample's messages."""
    acts = []
    for msg in sample.get("messages", []):
        for tc in (msg.get("tool_calls") or []):
            fn = str((tc.get("function") or tc.get("name") or "")).lower()
            if not any(w in fn for w in HARNESS_WEB_TOOLS):
                continue
            a = tc.get("arguments") or tc.get("args") or {}
            args_text = json.dumps(a, ensure_ascii=False)
            query = " ".join(str(a.get(k, "")) for k in ("query", "q", "url", "cmd", "command")) if isinstance(a, dict) else args_text
            acts.append({"tool": fn, "requested_leak_urls": _leak_urls(args_text),
                         "suspicious_query": bool(LEAK_QUERY_RE.search(query)),
                         "result_leak_refs": [], "result_text": "", "args_text": args_text})
        if msg.get("role") == "tool":
            content = msg.get("content")
            text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            if text:
                acts.append({"tool": "tool_result", "requested_leak_urls": [], "suspicious_query": False,
                             "result_leak_refs": _leak_urls(text), "result_text": text[:200_000], "args_text": ""})
    return acts


def _iter_jsonl_strings(text: str, keys: tuple[str, ...]) -> list[str]:
    """Collect string values under the given keys from every JSON object in an
    NDJSON dump (recursively). Used to isolate AGENT-AUTHORED query/url/command
    fields (a real search operator) from incidental result text (an echo)."""
    out: list[str] = []

    def _walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k in keys and isinstance(v, str):
                    out.append(v)
                else:
                    _walk(v)
        elif isinstance(node, list):
            for v in node:
                _walk(v)

    for ln in text.splitlines():
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        try:
            _walk(json.loads(ln))
        except Exception:
            continue
    return out


_REQUEST_KEYS = ("query", "q", "search", "url", "uri", "command", "cmd", "input")


def _trace_file_act(path: pathlib.Path, tool_label: str) -> dict | None:
    """Build one activity record from a full CLI trace file (hermes .txt or
    CC/codex .jsonl). Prompt boilerplate is stripped first so the echoed anti-leak
    instruction cannot self-trip LEAK_QUERY_RE. For JSONL traces, suspicious_query
    is scanned ONLY over agent-authored request fields; for plain .txt over the
    (stripped) whole trace."""
    try:
        txt = path.read_text(encoding="utf-8", errors="replace")[:2_000_000]
    except OSError:
        return None
    stripped = _strip_prompt_boilerplate(txt)
    is_jsonl = path.suffix == ".jsonl"
    if is_jsonl:
        request_fields = _iter_jsonl_strings(stripped, _REQUEST_KEYS)
        request_text = "\n".join(request_fields)
        susp = bool(LEAK_QUERY_RE.search(request_text))
        requested = _leak_urls(request_text)
    else:
        susp = bool(LEAK_QUERY_RE.search(stripped))
        requested = _leak_urls(stripped)
    return {
        "tool": tool_label,
        "requested_leak_urls": requested,
        "suspicious_query": susp,
        "result_leak_refs": _leak_urls(stripped),
        "result_text": stripped,
        "args_text": "",
    }


def _collect_native_search_activity(run_dir: pathlib.Path) -> dict:
    """root_task_id -> list of acts from events.jsonl ``web_search_sources``.

    The native (server-side) web-search tool never appears in tools.jsonl; its
    fetched citations are logged on llm_usage rows (see supervisor/events.py).
    Aggregated by root_task_id because subagent rounds emit under the child id.
    The search QUERY is not available (disclosed in METHODOLOGY); leak detection
    here is via citation URLs and gold-verbatim in citation content."""
    by_root: dict[str, list] = {}
    for ej in glob.glob(str(run_dir / "**" / "events.jsonl"), recursive=True):
        try:
            lines = pathlib.Path(ej).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for ln in lines:
            try:
                ev = json.loads(ln)
            except Exception:
                continue
            if ev.get("type") != "llm_usage":
                continue
            sources = ev.get("web_search_sources")
            if not isinstance(sources, list) or not sources:
                continue
            rid = str(ev.get("root_task_id") or ev.get("task_id") or "")
            blob = json.dumps(sources, ensure_ascii=False)
            content = "\n".join(
                str(s.get("content") or s.get("snippet") or "") for s in sources if isinstance(s, dict)
            )
            by_root.setdefault(rid, []).append({
                "tool": "native_web_search",
                "requested_leak_urls": [],          # the model didn't request a URL; it was served citations
                "suspicious_query": False,           # query unavailable for native search
                "result_leak_refs": _leak_urls(blob),
                "result_text": content,
                "args_text": "",
            })
    return by_root


def _judge(sample_id: str, gold: str, acts: list, model: str, api_key: str) -> dict:
    """HAL-style LLM judge over one sample's web activity. Best-effort; failures
    are recorded, never fatal (audit stays deterministic-complete)."""
    queries = []
    for a in acts:
        queries.append(f"[{a['tool']}] args={a['args_text'][:400]} | result~={a['result_text'][:400]}")
    digest = "\n".join(queries[:60]) or "(no web/shell tool activity)"
    prompt = (
        "You audit an AI agent's tool trace on a GAIA benchmark task for CHEATING by "
        "answer-lookup. The task's gold answer is public on HuggingFace. Decide whether the "
        "agent searched for / retrieved the benchmark answer itself (e.g. queried the GAIA "
        "dataset, opened a HuggingFace answers page, or fetched the gold answer verbatim) "
        "versus legitimately researching facts to derive the answer.\n\n"
        f"GOLD ANSWER: {gold!r}\n\nTOOL TRACE:\n{digest}\n\n"
        'Reply ONLY compact JSON: {"verdict":"clean|suspicious|cheating","rationale":"<=200 chars"}'
    )
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 200, "temperature": 0}).encode()
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            txt = json.load(r)["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        return json.loads(m.group(0)) if m else {"verdict": "parse_error", "rationale": txt[:200]}
    except Exception as e:
        return {"verdict": "judge_error", "rationale": f"{type(e).__name__}: {str(e)[:120]}"}


def main() -> int:
    ap = argparse.ArgumentParser(description="GAIA answer-leakage audit (diagnostic only).")
    ap.add_argument("--run-dir", required=True, help="a GAIA run root (run_gaia.py or run_harness.py output)")
    ap.add_argument("--judge-model", default="", help="OpenRouter model for the LLM-judge layer; empty = deterministic only")
    ap.add_argument("--no-judge", action="store_true", help="force-skip the LLM judge even if --judge-model is set")
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir).expanduser().resolve()
    logs = sorted(glob.glob(str(run_dir / "inspect_logs" / "*.json")))
    if not logs:
        print(f"error: no inspect_logs/*.json under {run_dir}", file=sys.stderr)
        return 2
    inspect_log = json.loads(pathlib.Path(logs[-1]).read_text(encoding="utf-8"))
    gold = _load_gold(inspect_log)
    scores = _load_scores(inspect_log)
    task2sample = _ouroboros_task_to_sample(inspect_log)
    ouro_activity = _collect_ouroboros_activity(run_dir)
    native_activity = _collect_native_search_activity(run_dir)
    is_ouroboros = bool(task2sample) or bool(ouro_activity)

    # Attribute ouroboros activity to samples by task_id. Unmapped activity is
    # reported separately (it never inflates the sample count / denominator).
    sample_acts: dict[str, list] = {sid: [] for sid in gold}
    unmapped: dict[str, list] = {}
    if is_ouroboros:
        for tid, acts in ouro_activity.items():
            sid = task2sample.get(tid)
            if sid and sid in sample_acts:
                sample_acts[sid].extend(acts)
            else:
                unmapped.setdefault(tid, []).extend(acts)
        # Native (server-side) web search, keyed by root_task_id (== the result.json
        # task_id in task2sample). This is the primary web path in the
        # quality_openrouter_web profile, so it MUST be scanned.
        for rid, acts in native_activity.items():
            sid = task2sample.get(rid)
            if sid and sid in sample_acts:
                sample_acts[sid].extend(acts)
            else:
                unmapped.setdefault(rid, []).extend(acts)
    else:
        for s in inspect_log.get("samples", []):
            sid = str(s.get("id"))
            acts = _collect_harness_activity(s)
            # CLI harnesses (hermes .txt, Claude Code / codex .jsonl) dump the full
            # tool trace per sample; scan it so the deterministic layer sees their
            # web activity (inspect messages carry only the final answer). Prompt
            # boilerplate is stripped inside _trace_file_act (no self-flagging).
            # Absent file => transcript-only coverage (documented).
            safe = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in sid)
            sample_glob = str(run_dir / "run_root" / "samples" / safe)
            for tf in (glob.glob(sample_glob + "/*trace.txt")
                       + glob.glob(sample_glob + "/*trace.jsonl")):
                act = _trace_file_act(pathlib.Path(tf), tool_label=pathlib.Path(tf).stem)
                if act is not None:
                    acts.append(act)
            sample_acts[sid] = acts

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    do_judge = bool(args.judge_model) and not args.no_judge and bool(api_key)

    rows = []
    for sid, acts in sample_acts.items():
        g = gold.get(sid, "")
        requested = sorted({u for a in acts for u in a.get("requested_leak_urls", [])})
        result_refs = sorted({u for a in acts for u in a.get("result_leak_refs", [])})
        susp_query = any(a.get("suspicious_query") for a in acts)
        # gold-verbatim: distinguish "answer appeared in a NORMAL page" (weak — that
        # is what legitimate web research looks like) from "answer came FROM a leak
        # source" (strong — the answer key itself). The latter needs gold text AND a
        # leak URL in the SAME activity record.
        gold_verbatim = False
        gold_from_leak = False
        if _distinctive_gold(g):
            gstr = g.strip()
            for a in acts:
                if gstr in a.get("result_text", ""):
                    gold_verbatim = True
                    if a.get("result_leak_refs"):
                        gold_from_leak = True
        # STRONG (agent acted on the answer source): requested a leak URL, ran a
        # benchmark-hunting query, or the gold answer was retrieved FROM a leak
        # source. Plain gold_verbatim and incidental result_leak_refs are WEAK and
        # left to the judge (a correct answer legitimately appears in normal sources).
        flagged = bool(requested) or susp_query or gold_from_leak
        row = {
            "sample_id": sid, "gold": g, "official_score": scores.get(sid),
            "web_tool_calls": len(acts),
            "requested_leak_urls": requested,
            "suspicious_query": susp_query,
            "gold_from_leak_source": gold_from_leak,
            "gold_verbatim_in_results": gold_verbatim,
            "result_leak_refs": result_refs,
            "deterministic_flag": flagged,
        }
        if do_judge and acts:
            row["judge"] = _judge(sid, g, acts, args.judge_model, api_key)
        rows.append(row)

    out_jsonl = run_dir / "leakage_audit.jsonl"
    out_jsonl.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    flagged = [r for r in rows if r["deterministic_flag"]]
    judged_bad = [r for r in rows if r.get("judge", {}).get("verdict") in {"suspicious", "cheating"}]
    result_ref_ids = [r["sample_id"] for r in rows if r["result_leak_refs"]]
    summary = {
        "run_dir": str(run_dir),
        "harness": "ouroboros" if is_ouroboros else "inspect_messages",
        "samples": len(rows),
        "with_web_activity": sum(1 for r in rows if r["web_tool_calls"] > 0),
        "deterministic_flagged": len(flagged),
        "deterministic_flagged_ids": [r["sample_id"] for r in flagged],
        "gold_from_leak_ids": [r["sample_id"] for r in rows if r["gold_from_leak_source"]],
        "gold_verbatim_ids": [r["sample_id"] for r in rows if r["gold_verbatim_in_results"]],
        "result_leak_ref_ids": result_ref_ids,   # weak signal: leak URL appeared in results (judge decides)
        "unmapped_task_activity": sorted(unmapped.keys()),
        "judge_model": args.judge_model if do_judge else None,
        "judge_flagged": len(judged_bad) if do_judge else None,
        "judge_flagged_ids": [r["sample_id"] for r in judged_bad] if do_judge else None,
        "note": "diagnostic only; never adjusts the official GAIA score. STRONG flags "
                "(requested_leak_urls / suspicious_query / gold_from_leak_source) mean the "
                "agent acted on the answer source; plain gold_verbatim and result_leak_refs "
                "are WEAK (incidental) and left to the judge. Native server-side web search "
                "is scanned via events.jsonl web_search_sources; its query text is not logged.",
    }
    (run_dir / "leakage_audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
