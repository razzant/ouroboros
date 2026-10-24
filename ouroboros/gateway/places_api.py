"""Place choices through the gateway's authenticated owner surface."""
from __future__ import annotations

import asyncio
from starlette.responses import JSONResponse
from starlette.routing import Route
from ouroboros.gateway._helpers import request_drive_root, run_sync_to_completion
from ouroboros.places import AUTHOR_OWNER, PlaceError, edit_place, place_room_index
from ouroboros.place_projection import connect_declared_place, place_view, places_view
from ouroboros.store import StoreError


def update_place(root, place_id: str, body: dict) -> dict:
    allowed = {"state", "wakes", "add_note", "remove_note", "working_folder", "model",
               "confirm_owner_account", "revoke_owner_account", "expected_revision"}
    if not isinstance(body, dict) or not body or set(body) - allowed:
        raise PlaceError("place_update_invalid")
    actions = []
    mapping = {"state": ("set_state", "state"), "wakes": ("set_wakes", "wakes"),
               "add_note": ("add_note", "text"), "remove_note": ("remove_note", "note_id"),
               "working_folder": ("set_working_folder", "working_folder"), "model": ("set_model", "model"),
               "revoke_owner_account": ("revoke_owner_account", "platform_actor_id")}
    for field, value in body.items():
        if field == "expected_revision":
            continue
        if field == "confirm_owner_account":
            if not isinstance(value, dict) or set(value) - {"platform_actor_id", "label", "account_id"}:
                raise PlaceError("place_account_invalid")
            actions.append((field, value))
        else:
            action, key = mapping[field]
            actions.append((action, {key: value}))
    saved = edit_place(root, place_id, "update", author=AUTHOR_OWNER, changes=actions,
                       expected_revision=body.get("expected_revision"))
    return {"ok": True, "place": place_view(root, saved)}


async def api_places(request):
    root = request_drive_root(request)
    place_id = str(request.path_params.get("place_id") or "")
    try:
        if request.method == "GET":
            return JSONResponse(await asyncio.to_thread(places_view, root))
        if request.method == "DELETE":
            await run_sync_to_completion(edit_place, root, place_id, "disconnect", author=AUTHOR_OWNER)
            return JSONResponse({"ok": True})
        body = await request.json()
        if place_id:
            return JSONResponse(await run_sync_to_completion(update_place, root, place_id, body))
        if not isinstance(body, dict) or set(body) - {"connector", "account_id", "conversation_id"}:
            raise PlaceError("place_connect_invalid")
        if not isinstance(body.get("connector"), str):
            raise PlaceError("place_connect_invalid", "connector")
        def connect():
            return {"ok": True, "place": place_view(root, connect_declared_place(root, **body))}
        return JSONResponse(await run_sync_to_completion(connect))
    except (ValueError, TypeError) as exc:
        return JSONResponse({"ok": False, "error": str(exc), "code": getattr(exc, "code", "place_invalid")},
                            status_code=409 if getattr(exc, "code", "") == "place_revision_conflict" else 400)


async def api_place_rooms(request):
    """Lazy owner read of captured rooms, not remote-provider history or task admission."""
    try:
        place_id = str(request.path_params['place_id'])
        rooms = await asyncio.to_thread(place_room_index, request_drive_root(request), place_id)
        return JSONResponse({'place_id': place_id, 'rooms': rooms}, headers={'Cache-Control': 'no-store'})
    except PlaceError as exc:
        return JSONResponse({'ok': False, 'code': exc.code, 'error': str(exc)},
                            status_code=404 if exc.code == 'place_not_found' else 503)
    except (OSError, ValueError, StoreError) as exc:
        return JSONResponse({'ok': False, 'code': 'place_rooms_unavailable', 'error': str(exc)}, status_code=503)


def places_routes():
    return [Route("/api/places", api_places, methods=["GET", "POST"]),
            Route("/api/places/{place_id}/rooms", api_place_rooms, methods=["GET"]),
            Route("/api/places/{place_id}", api_places, methods=["POST", "DELETE"])]
