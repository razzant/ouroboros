# tests/golden_traces — golden-trace harness for ToolRegistry.execute.
#
# Captures the ORDER and OBSERVABLE ARGUMENTS of the key dispatch-pipeline
# guard calls plus the final returned text, normalized for tmp paths, so a
# dispatch refactor can be proven behavior-identical against the recorded
# fixtures (tests/golden_traces/fixtures/*.json).
