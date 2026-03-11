"""
Ouroboros — Knowledge Router.

Lightweight keyword-based routing of knowledge base topics to tasks.
Automatically surfaces relevant knowledge in context without requiring
the agent to explicitly call knowledge_read.

NOT embedding-based (Bible P5: minimalism — no vector DB dependency).
Uses topic-to-keyword mapping maintained in knowledge file headers.

Bible alignment:
  P3 (LLM-First): Knowledge is injected as text, LLM decides relevance.
  P5 (Minimalism): Keyword matching, no external dependencies.
  P6 (Becoming): Accumulated wisdom is automatically applied.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# Pattern to match keyword headers in knowledge files
# Supports: <!-- keywords: browser, playwright, screenshot -->
#           <!-- keywords: git, push, commit, branch -->
_KEYWORD_RE = re.compile(
    r'<!--\s*keywords?:\s*(.+?)\s*-->', re.IGNORECASE
)

# Fallback: extract keywords from topic filename
# e.g., "browser-gotchas" → ["browser", "gotchas"]
_FILENAME_SPLIT = re.compile(r'[-_.]')

# Maximum characters to inject per topic (prevent context bloat)
MAX_TOPIC_CHARS = 3000

# Maximum topics to inject per task
MAX_INJECTED_TOPICS = 3


def _extract_keywords_from_file(path: Path) -> List[str]:
    """Extract keywords from a knowledge file.

    Checks for explicit keyword header first, falls back to filename.
    """
    keywords = []

    # Try explicit header
    try:
        # Read only first 500 chars — keywords should be at the top
        head = path.read_text(encoding="utf-8")[:500]
        match = _KEYWORD_RE.search(head)
        if match:
            raw = match.group(1)
            keywords = [k.strip().lower() for k in raw.split(",") if k.strip()]
    except Exception:
        log.debug("Failed to read keywords from %s", path, exc_info=True)

    # Fallback: derive from filename
    if not keywords:
        stem = path.stem.lower()
        if stem.startswith("_"):
            return []  # Skip index files
        keywords = [w for w in _FILENAME_SPLIT.split(stem) if len(w) > 2]

    return keywords


def build_keyword_index(knowledge_dir: Path) -> Dict[str, List[str]]:
    """Build a mapping of topic_name → keywords for all knowledge files.

    Returns:
        Dict mapping topic name (filename stem) to list of keywords.
    """
    index: Dict[str, List[str]] = {}

    if not knowledge_dir.exists():
        return index

    for md_file in knowledge_dir.glob("*.md"):
        if md_file.name.startswith("_"):
            continue
        topic = md_file.stem
        keywords = _extract_keywords_from_file(md_file)
        if keywords:
            index[topic] = keywords

    return index


def route_knowledge(
    task_text: str,
    knowledge_dir: Path,
    max_topics: int = MAX_INJECTED_TOPICS,
) -> List[Tuple[str, float]]:
    """Find knowledge topics relevant to a task description.

    Args:
        task_text: The task text / user message to match against.
        knowledge_dir: Path to memory/knowledge/ directory.
        max_topics: Maximum number of topics to return.

    Returns:
        List of (topic_name, score) tuples, sorted by relevance.
        Score is normalized 0.0-1.0 based on keyword match density.
    """
    if not task_text or not knowledge_dir.exists():
        return []

    task_lower = task_text.lower()
    # Tokenize task text for word-boundary matching
    task_words = set(re.findall(r'\b\w+\b', task_lower))

    index = build_keyword_index(knowledge_dir)
    if not index:
        return []

    scored: List[Tuple[str, float]] = []

    for topic, keywords in index.items():
        if not keywords:
            continue

        # Score: fraction of topic keywords found in task text
        # Use both substring matching (for multi-word terms) and word matching
        matches = 0
        for kw in keywords:
            # Exact word match (higher signal)
            if kw in task_words:
                matches += 1.0
            # Substring match (lower signal, catches compound terms)
            elif kw in task_lower:
                matches += 0.5

        if matches > 0:
            # Normalize by number of keywords (topics with fewer keywords
            # need fewer matches to be considered relevant)
            score = matches / len(keywords)
            scored.append((topic, round(score, 3)))

    # Sort by score descending, then alphabetically for ties
    scored.sort(key=lambda x: (-x[1], x[0]))

    return scored[:max_topics]


def load_relevant_knowledge(
    task_text: str,
    knowledge_dir: Path,
    max_topics: int = MAX_INJECTED_TOPICS,
    max_chars_per_topic: int = MAX_TOPIC_CHARS,
) -> Optional[str]:
    """Load and format relevant knowledge for injection into context.

    Returns a formatted string ready to append to the dynamic context section,
    or None if no relevant knowledge was found.

    Integration point: Called from context.py's _build_recent_sections() or
    build_llm_messages() when assembling the dynamic text block.
    """
    matches = route_knowledge(task_text, knowledge_dir, max_topics)

    if not matches:
        return None

    sections = []

    for topic, score in matches:
        topic_path = knowledge_dir / f"{topic}.md"
        if not topic_path.exists():
            continue

        try:
            content = topic_path.read_text(encoding="utf-8")

            # Strip the keyword header (don't waste context on metadata)
            content = _KEYWORD_RE.sub("", content).strip()

            # Truncate if needed
            if len(content) > max_chars_per_topic:
                content = content[:max_chars_per_topic] + "\n... (truncated)"

            sections.append(f"### {topic} (relevance: {score:.1f})\n\n{content}")
        except Exception:
            log.debug("Failed to load knowledge topic %s", topic, exc_info=True)
            continue

    if not sections:
        return None

    header = f"## Relevant Knowledge ({len(sections)} topic{'s' if len(sections) != 1 else ''})\n\n"
    return header + "\n\n---\n\n".join(sections)
