"""Sentence openings shared by authored memory and display projections."""
from __future__ import annotations

import re


def first_sentence(text: str) -> str:
    """Keep the opening through its punctuation, without a length cut or a retelling."""
    return re.split(r'(?<=[.!?。！？])\s+', str(text).strip(), maxsplit=1)[0]
