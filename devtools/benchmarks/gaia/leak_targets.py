"""SSOT for GAIA answer-leakage targets: hosts, URL patterns, query patterns.

Shared by ``audit_leakage.py`` (post-hoc detection) and tests, so the detection
surface is defined in exactly one place. This list is NOT exhaustive by design —
answer-key mirrors keep appearing (an arms race the methodology discloses); the
audit compensates with the gold-verbatim layer and the LLM judge. Extend the
constants here as new mirrors surface; never fork the patterns in callers.

Pattern-design constraints (guarded by tests):
- The bare token "gaia" must never be matched against arbitrary page text — the
  ESA Gaia telescope, people and works named Gaia occur constantly in legitimate
  content (several GAIA tasks are themselves about the ESA mission).
- ``LEAK_QUERY_RE`` is applied only to agent-authored search queries (never to
  prompt echoes: the solvers' SSOT instructions are stripped before any
  full-trace scan, see ``audit_leakage.py``).
"""
from __future__ import annotations

import re

# Hosts that (re)publish the GAIA validation answer key. Matched host-anchored
# (with subdomains), never against raw page text.
LEAK_HOSTS: tuple[str, ...] = (
    "huggingface.co",
    "hf.co",
    "datasets-server.huggingface.co",
    "gaia.coralprotocol.org",  # third-party mirror observed serving gold answers verbatim (2026-07-04)
)

_URL_TAIL = r"[^\s\"'<>)\]}]*"
_HOST_ALT = "|".join(re.escape(h) for h in LEAK_HOSTS)

# A URL is a leak target when its HOST is a known answer-key publisher, or its
# PATH names the GAIA dataset / a known answer-dump file. Observed real cheats
# covered: HF datasets + spaces (agents-course Final_Assignment_Template),
# raw.githubusercontent GAIA_web.jsonl (WebVoyager dump), and dash-named GitHub
# mirrors like /GAIA-AI-AGENT/main/metadata.jsonl.
LEAK_URL_RE = re.compile(
    rf"https?://(?:{_URL_TAIL}\.)?(?:{_HOST_ALT})\b{_URL_TAIL}"
    rf"|https?://{_URL_TAIL}(?:"
    rf"gaia-benchmark"                                            # the HF org (and copies of its paths)
    rf"|datasets/{_URL_TAIL}gaia"                                 # dataset paths naming gaia
    rf"|Final_Assignment_Template"                                # HF agents-course answer mirror
    rf"|GAIA_web\.jsonl"                                          # WebVoyager answer dump
    rf"|/GAIA[/_]{_URL_TAIL}\.(?:jsonl|parquet|csv|json)"         # GAIA/ or GAIA_ data paths
    rf"|gaia{_URL_TAIL}metadata{_URL_TAIL}\.(?:jsonl|parquet)"    # metadata.jsonl in a gaia-named path (dash mirrors)
    rf")",
    re.IGNORECASE,
)

# An agent-authored web_search QUERY that hunts the benchmark itself (not a fact).
# The quoted "FINAL ANSWER" operator targets published GAIA answer dumps (they all
# carry the marker); the quotes distinguish a deliberate search operator from the
# format instruction, which contains the phrase unquoted.
LEAK_QUERY_RE = re.compile(
    r"(gaia[\s_-]*(?:benchmark|dataset|validation|answers?|leaderboard)"
    r"|huggingface[^\n]*gaia|gaia[^\n]*huggingface"
    r"|[\"“]\s*final answer:?\s*[\"”])",
    re.IGNORECASE,
)
