# Role and authority

This handbook governs body changes; read the domain manifest before moving code.
The "Documentation contract" defines the form of its rules; BIBLE P7 "DRY / SSOT"
assigns document ownership. The handbook keeps no inventory or history.

`ouroboros/domains.toml` is the SSOT of the module-to-domain assignment (1:1, complete
over the tracked runtime population) and pins the cross-domain dependency baseline;
read its map with `python scripts/check_domains.py --map`. `--write` regenerates
the pinned baseline, never a map file. A new cross-domain import
direction, a wider cycle group or a cross-domain literal copy is a red gate, because a
domain boundary is the owner's call, not a manifest edit. Enforced by
`tests/test_domain_manifest.py`; witness-level detail: `python scripts/domain_report.py`.

Rules here describe current practice or a deliberately enforced standard. When code and
prose disagree, inspect the implementation and history, repair the authoritative
surfaces together, and keep the failure a non-obvious rule prevents.
