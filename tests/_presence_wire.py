"""The connector's ``POST /presence/turn`` event as synthetic tests send it (CREATING_SKILLS "Places and focus
transport ingress"). ``PresenceTurnEvent`` also holds what the Host itself admits (negotiated versions, the
direct-reply route); selecting these keys sends none of those, however many the Host adds."""

# ``identity`` is left out: a connector sends it only after negotiating ``presence_identity``; a test adds it.
WIRE_KEYS = frozenset({"source_event_id", "provider", "account_id", "conversation_id", "thread_id",
                       "conversation_key", "actor", "conversation", "message", "text", "facts"})
