"""Inspect-evals solver wrapper for GAIA."""

# SSOT for the GAIA answer-format instruction (the official benchmark expects a
# format/prefix prompt). Shared by every solver so the three harnesses stay in
# parity. GAIA's quasi-exact-match scorer normalizes case/punctuation/articles but
# NOT scale or wording, so a clear format instruction is the honest, methodology-
# sanctioned way to align the agent's own answer shape. Adapter/prompt only — never
# imported into runtime core (core normalization would hurt ordinary users).
GAIA_FORMAT_INSTRUCTION = (
    "\n\nWork through the task, then end your response with a single line, "
    "exactly: FINAL ANSWER: <your answer>\nThe answer must be a number or as few "
    "words as possible, with no units unless asked."
)

# SSOT for the anti-answer-lookup rule, appended by every solver alongside the
# format instruction so all harnesses run under the identical task contract.
# Added after the 2026-07-04 runs caught agents googling the published answer
# key instead of solving (see METHODOLOGY.md "Answer-leakage audit protocol").
# Wording constraints (guarded by tests): must NOT contain the benchmark name
# (would prime the model toward the answer source and self-trip the audit's
# LEAK_QUERY_RE when traces echo the prompt) and must NOT contain the literal
# "FINAL ANSWER" (solvers gate the format instruction on that substring, and
# the audit treats it as an answer-hunting query marker).
GAIA_ANTI_LEAK_INSTRUCTION = (
    "\n\nIMPORTANT - answer integrity: This question comes from a public evaluation "
    "set whose reference answers are published on the internet. Solve the task from "
    "primary sources only. Do NOT search for or open the evaluation set itself, its "
    "answer/metadata files, dataset mirrors, leaderboards, or third-party write-ups "
    "that reveal the expected answer. Looking up the reference answer instead of "
    "deriving it is a protocol violation and invalidates your answer."
)
