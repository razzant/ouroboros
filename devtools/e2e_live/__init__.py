"""Live E2E stand: K isolated real Ouroboros servers, one owner-shaped scenario each.

``python -m devtools.e2e_live.run_live_lanes --stub --lanes 2`` proves the plumbing
against the loopback stub model for $0; the same command without ``--stub`` and with a
key in ``$OUROBOROS_E2E_LIVE_OPENROUTER_KEY`` is the paid run. Operator code outside the
runtime import graph (DEVELOPMENT "Devtools isolation").
"""
