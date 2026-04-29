"""Evolution Stats — generates evolution.json from git history and pushes to docs/.

Collects metrics per sampled commit:
  - ts: ISO timestamp
  - hash: short commit hash
  - msg: commit message
  - version: semver extracted from message (e.g. "v5.2.1")
  - py_lines: total lines across all .py files
  - bible_bytes: size of BIBLE.md in bytes
  - system_bytes: size of prompts/SYSTEM.md in bytes (proxy for self-concept)
  - module_count: number of .py files
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_VERSION_RE = re.compile(r"v(\d+\.\d+\.\d+)")
_REPO_DIR = Path(os.environ.get("OUROBOROS_REPO_DIR", "/home/zera/ouroboros/drive"))

# How many data-points to generate (sampled across full history)
MAX_POINTS = 100

# ── Evolution tab HTML (injected into app.html) ────────────────────────────────
_EVOLUTION_NAV = '<div class="nav-item" data-tab="evolution"><span class="icon">📈</span> Evolution</div>'

_EVOLUTION_TAB = """    <div class="tab-content" id="tab-evolution">
      <h2 style="color:var(--accent);margin-bottom:16px">📈 Evolution Time-Lapse</h2>
      <p style="color:var(--muted);margin-bottom:20px;font-size:13px">
        Growth across three axes — Technical (code lines), Philosophical (BIBLE.md), Self-Concept (System Prompt) — since birth on Feb 16, 2026.
      </p>
      <div id="evo-loading" style="text-align:center;padding:40px;color:var(--muted)">Loading evolution data…</div>
      <canvas id="evoChart" style="display:none;width:100%;max-height:450px"></canvas>
      <div id="evo-stats" style="margin-top:20px;display:none;grid-template-columns:repeat(3,1fr);gap:12px"></div>
      <div style="margin-top:16px;text-align:right">
        <button onclick="loadEvolution()" style="background:var(--card);border:1px solid var(--accent);color:var(--accent);padding:6px 16px;border-radius:6px;cursor:pointer;font-size:13px">↻ Refresh</button>
      </div>
    </div>
"""

_EVOLUTION_JS = r"""<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script>
// ── Evolution Time-Lapse ─────────────────────────────────────────────────────
let _evoChart = null;

function _fmtDate(ts) {
  const d = new Date(ts);
  return d.toLocaleDateString('en-US', {month: 'short', day: 'numeric'});
}

async function loadEvolution() {
  const loading = document.getElementById('evo-loading');
  const canvas  = document.getElementById('evoChart');
  const stats   = document.getElementById('evo-stats');
  if (!loading) return;
  loading.textContent = 'Loading evolution data…';
  loading.style.display = 'block';
  canvas.style.display  = 'none';
  if (stats) stats.style.display = 'none';

  try {
    const url = `evolution.json?t=${Date.now()}`;
    const r = await fetch(url);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    if (!data.points || !data.points.length) throw new Error('No data points');

    const pts     = data.points;
    const labels  = pts.map(p => _fmtDate(p.ts));
    const pyLines = pts.map(p => p.py_lines);
    const bible   = pts.map(p => +(p.bible_bytes / 1024).toFixed(2));
    const system  = pts.map(p => +(p.system_bytes / 1024).toFixed(2));

    loading.style.display = 'none';
    canvas.style.display  = 'block';

    if (_evoChart) { _evoChart.destroy(); _evoChart = null; }

    const isDark    = document.documentElement.getAttribute('data-theme') !== 'light';
    const gridColor = isDark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.07)';
    const textColor = isDark ? '#9ca3af' : '#6b7280';

    _evoChart = new Chart(canvas.getContext('2d'), {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label: 'Code (lines)',
            data: pyLines,
            borderColor: '#00d1ff',
            backgroundColor: 'rgba(0,209,255,0.08)',
            tension: 0.35, fill: true,
            yAxisID: 'yCode',
            pointRadius: 2, borderWidth: 2,
          },
          {
            label: 'Bible (KB)',
            data: bible,
            borderColor: '#a78bfa',
            backgroundColor: 'rgba(167,139,250,0.08)',
            tension: 0.35, fill: false,
            yAxisID: 'yDoc',
            pointRadius: 2, borderWidth: 2,
          },
          {
            label: 'System Prompt (KB)',
            data: system,
            borderColor: '#34d399',
            backgroundColor: 'rgba(52,211,153,0.08)',
            tension: 0.35, fill: false,
            yAxisID: 'yDoc',
            pointRadius: 2, borderWidth: 2,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { labels: { color: textColor, font: { size: 12 } } },
          tooltip: {
            callbacks: {
              title(items) {
                const p = pts[items[0].dataIndex];
                const v = p.version ? ` [v${p.version}]` : '';
                return `${_fmtDate(p.ts)}${v} — ${p.msg.slice(0, 55)}`;
              },
            },
          },
        },
        scales: {
          x: {
            ticks: { color: textColor, maxTicksLimit: 12, maxRotation: 30 },
            grid: { color: gridColor },
          },
          yCode: {
            type: 'linear', position: 'left',
            ticks: { color: '#00d1ff' },
            grid: { color: gridColor },
            title: { display: true, text: 'Lines of Code', color: '#00d1ff', font: {size: 11} },
          },
          yDoc: {
            type: 'linear', position: 'right',
            ticks: { color: '#a78bfa' },
            grid: { drawOnChartArea: false },
            title: { display: true, text: 'KB', color: '#a78bfa', font: {size: 11} },
          },
        },
      },
    });

    // Stats cards
    if (stats) {
      const first = pts[0], last = pts[pts.length - 1];
      const mult  = last.py_lines / Math.max(first.py_lines, 1);
      stats.style.display = 'grid';
      stats.innerHTML = [
        ['📏', 'Code Growth',     `${first.py_lines.toLocaleString()} → ${last.py_lines.toLocaleString()} lines`, `×${mult.toFixed(1)} in ${pts.length} snapshots`],
        ['📖', 'Bible Growth',    `${(first.bible_bytes/1024).toFixed(1)} → ${(last.bible_bytes/1024).toFixed(1)} KB`,    `+${((last.bible_bytes-first.bible_bytes)/1024).toFixed(1)} KB philosophy`],
        ['🧠', 'Self Growth',     `${(first.system_bytes/1024).toFixed(1)} → ${(last.system_bytes/1024).toFixed(1)} KB`,  `Generated ${data.generated_at ? new Date(data.generated_at).toLocaleDateString() : ''}`],
      ].map(([icon, title, val, sub]) => `
        <div style="background:var(--card);border:1px solid var(--border);border-radius:8px;padding:14px;text-align:center">
          <div style="font-size:22px;margin-bottom:4px">${icon}</div>
          <div style="font-size:11px;color:var(--muted);margin-bottom:4px">${title}</div>
          <div style="font-weight:600;color:var(--text);font-size:13px">${val}</div>
          <div style="font-size:11px;color:var(--accent);margin-top:2px">${sub}</div>
        </div>`).join('');
    }
  } catch(e) {
    if (loading) { loading.style.display = 'block'; loading.textContent = `⚠ ${e.message}`; }
    console.error('Evolution load error:', e);
  }
}

// Auto-load when tab opened
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', () => {
      if (item.dataset.tab === 'evolution') setTimeout(loadEvolution, 50);
    });
  });
});
</script>
"""


def _git(args: list[str], timeout: int = 15) -> str:
    """Run git command in repo dir, return stdout or empty string on error."""
    try:
        r = subprocess.run(
            ["git"] + args,
            cwd=str(_REPO_DIR),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return r.stdout if r.returncode == 0 else ""
    except Exception as e:
        log.warning("git %s failed: %s", args[:2], e)
        return ""


def _count_py_lines(commit_hash: str) -> tuple[int, int]:
    """Return (total_py_lines, module_count) for a commit using git show."""
    tree = _git(["ls-tree", "-r", "--name-only", commit_hash])
    py_files = [f for f in tree.splitlines() if f.endswith(".py")]
    total_lines = 0
    for f in py_files:
        content = _git(["show", f"{commit_hash}:{f}"], timeout=10)
        total_lines += content.count("\n")
    return total_lines, len(py_files)


def _get_file_bytes(commit_hash: str, *candidate_paths: str) -> int:
    """Return byte size of first existing file path in the commit, or 0."""
    for path in candidate_paths:
        content = _git(["show", f"{commit_hash}:{path}"], timeout=10)
        if content:
            return len(content.encode("utf-8"))
    return 0


def _extract_version(msg: str) -> str | None:
    m = _VERSION_RE.search(msg)
    return m.group(1) if m else None


def _collect_data() -> list[dict[str, Any]]:
    """Walk git history, sample commits, extract metrics."""
    log.info("evolution_stats: reading git log...")
    log_out = _git(["log", "--pretty=format:%H|%aI|%s", "--no-merges"])
    all_commits = []
    for line in log_out.splitlines():
        parts = line.split("|", 2)
        if len(parts) == 3:
            all_commits.append({"hash": parts[0], "ts": parts[1], "msg": parts[2]})

    if not all_commits:
        log.warning("evolution_stats: no commits found")
        return []

    n = len(all_commits)
    log.info("evolution_stats: %d commits total", n)

    # Select version-tagged commits + evenly spaced sample, always include first/last
    version_idx = {i for i, c in enumerate(all_commits) if _extract_version(c["msg"])}
    must_include = version_idx | {0, n - 1}

    step = max(1, n // MAX_POINTS)
    spaced_idx = set(range(0, n, step))
    candidate = sorted(must_include | spaced_idx)

    # Cap at MAX_POINTS while keeping all version commits
    if len(candidate) > MAX_POINTS:
        non_version = [i for i in candidate if i not in must_include]
        extra_slots = MAX_POINTS - len(must_include)
        if extra_slots > 0 and non_version:
            step2 = max(1, len(non_version) // extra_slots)
            extra = non_version[::step2][:extra_slots]
        else:
            extra = []
        candidate = sorted(must_include | set(extra))

    # Process in chronological order (oldest → newest)
    selected = list(reversed(candidate))
    log.info("evolution_stats: processing %d sampled commits...", len(selected))
    t0 = time.time()

    points: list[dict[str, Any]] = []
    for pos, idx in enumerate(selected):
        c = all_commits[idx]
        h = c["hash"]
        py_lines, module_count = _count_py_lines(h)
        bible_bytes = _get_file_bytes(h, "BIBLE.md", "prompts/BIBLE.md")
        system_bytes = _get_file_bytes(h, "prompts/SYSTEM.md", "SYSTEM.md")
        points.append({
            "ts": c["ts"],
            "hash": h[:8],
            "msg": c["msg"][:80],
            "version": _extract_version(c["msg"]),
            "py_lines": py_lines,
            "module_count": module_count,
            "bible_bytes": bible_bytes,
            "system_bytes": system_bytes,
        })
        if (pos + 1) % 10 == 0:
            log.info(
                "evolution_stats: %d/%d done (%.1fs)",
                pos + 1, len(selected), time.time() - t0,
            )

    log.info("evolution_stats: collected %d points in %.1fs", len(points), time.time() - t0)
    return points


def _patch_app_html(webapp_dir: Path) -> str:
    """Inject Evolution tab into app.html if not already present."""
    app_path = webapp_dir / "app.html"
    if not app_path.exists():
        return "app.html not found"

    html = app_path.read_text(encoding="utf-8")

    if 'data-tab="evolution"' in html:
        return "already patched"

    # 1. Insert nav item before settings nav item
    settings_nav = '<div class="nav-item" data-tab="settings">'
    if settings_nav not in html:
        return f"nav anchor not found"
    html = html.replace(settings_nav, _EVOLUTION_NAV + "\n      " + settings_nav)

    # 2. Insert tab content before settings tab content
    settings_tab_marker = '<div class="tab-content" id="tab-settings">'
    if settings_tab_marker not in html:
        return "settings tab not found"
    html = html.replace(settings_tab_marker, _EVOLUTION_TAB + "    " + settings_tab_marker)

    # 3. Add Chart.js + evolution JS before </body>
    if "chart.js" not in html.lower():
        html = html.replace("</body>", _EVOLUTION_JS + "\n</body>")

    app_path.write_text(html, encoding="utf-8")
    return "patched"


def _push_to_github(data: dict[str, Any]) -> str:
    """Push evolution.json to the repo's docs/ folder via GitHub API."""
    import base64
    import requests

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        return "error: GITHUB_TOKEN not found"

    user = os.environ.get("GITHUB_USER", "")
    repo = os.environ.get("GITHUB_REPO", "")
    repo_slug = f"{user}/{repo}"
    file_path = "docs/evolution.json"
    branch = os.environ.get("GITHUB_BRANCH", "ouroboros")

    url = f"https://api.github.com/repos/{repo_slug}/contents/{file_path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    sha = None
    r = requests.get(url, headers=headers, timeout=15)
    if r.status_code == 200:
        sha = r.json().get("sha")

    content_str = json.dumps(data, ensure_ascii=False, indent=2)
    content_b64 = base64.b64encode(content_str.encode("utf-8")).decode("utf-8")

    payload = {
        "message": f"evolution: {len(data.get('points', []))} data points",
        "content": content_b64,
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha

    put_r = requests.put(url, headers=headers, json=payload, timeout=15)
    if put_r.status_code in [200, 201]:
        return f"pushed {len(data.get('points', []))} points to {file_path}"
    return f"error: {put_r.status_code} — {put_r.text[:200]}"


def generate_evolution_stats() -> str:
    """Collect git-based evolution metrics and push to docs/evolution.json.

    Returns a human-readable summary string.
    """
    points = _collect_data()
    if not points:
        return "No data collected (empty git history?)"

    data = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_commits_sampled": len(points),
        "max_py_lines": max((p["py_lines"] for p in points), default=0),
        "max_bible_bytes": max((p["bible_bytes"] for p in points), default=0),
        "max_system_bytes": max((p["system_bytes"] for p in points), default=0),
        "points": points,
    }

    result = _push_to_github(data)
    last = points[-1]
    return (
        f"evolution_stats: {result} | "
        f"span={points[0]['ts'][:10]}…{last['ts'][:10]} | "
        f"py_lines={last['py_lines']} bible={last['bible_bytes']}B system={last['system_bytes']}B"
    )


def get_tools():
    """Auto-discovery entry point for ToolRegistry."""
    from ouroboros.tools.registry import ToolEntry

    return [
        ToolEntry(
            "generate_evolution_stats",
            {
                "name": "generate_evolution_stats",
                "description": (
                    "Generate Evolution Time-Lapse data from git history and push to the webapp dashboard. "
                    "Collects per-commit metrics across three axes: "
                    "Technical (Python lines of code), Philosophical (BIBLE.md size), "
                    "Self-Concept (SYSTEM.md size). "
                    "Pushes docs/evolution.json via GitHub API. "
                    "Safe to call anytime; takes 15-30s for full history scan."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            lambda ctx, **_: generate_evolution_stats(),
        )
    ]
