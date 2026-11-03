"""Real-data stand: a candidate body booted on a contained copy of an install's data root.

``python -m devtools.real_data.stand snapshot|arm`` (README.md beside this file). Operator code
outside the runtime package: runtime modules never import it, and it never writes the live root.
"""
