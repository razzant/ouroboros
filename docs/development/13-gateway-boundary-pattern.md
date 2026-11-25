# Gateway Boundary Pattern

This chapter owns dependency direction at the browser boundary: inbound routes enter through `ouroboros/gateway/`, outbound provider and harness adapters live in `ouroboros/gateways/` and carry no domain policy, and every interface, the builtin one included, is a client of one published contract (why the facade exists: ARCHITECTURE §1 "Gateway Boundary v1").

Frontend calls route through `web/modules/api_client.js`. Never promote `gateway/host_service.py` callbacks into a general owner/task API, and do not require classes where established function owners preserve the boundary (CHECKLISTS item 23 (`gateway_parity`); `tests/test_gateway_parity.py`).

Publish routes/WS types via `gateway/endpoint_index.py`, `HTTP_SHAPES`/`WS_SHAPES`, contract TypedDicts, and field-identical JSDoc twins driving `/api/contract`; update client-visible meaning in CLIENT_CONTRACT with code. Applied `ServedUI` alone selects optional setup/login pages (auth and settings keep their writers); builtin recovery assets and the author kit stay builtin without mixing fallback files (`tests/test_client_contract_coverage.py`, `tests/test_ui_serving.py`).

Named skill chat ingress writes through the message-bus canonical writer pre-enqueue and correlates reads and cancel by `operation_ref` (custody, replay and the 409 rules: ARCHITECTURE §12). Cancel and intent writes match the read installation root; reads and cancel verify the complete source against actual task or direct-turn ownership, treating presentation annotations as discovery hints only (`tests/test_host_service_operation_identity.py`, `tests/test_host_service_operations.py`).

Dialogue storage changes preserve the public projection: `tests/test_dialogue_reader_golden.py` checks history, log tails, task events, and memory recall against `tests/fixtures/dialogue_readers.json` (normalizing only paths, witnesses, and snapshot token).
