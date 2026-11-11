"""Place and circle choices through the gateway's authenticated owner surface: every write here is my human's."""
from __future__ import annotations

import asyncio
from starlette.responses import JSONResponse
from starlette.routing import Route
from ouroboros.gateway._helpers import request_drive_root, run_sync_to_completion
from ouroboros.places import (AUTHOR_OWNER, PlaceError, create_circle, delete_circle, edit_circle, edit_place,
                              get_place, place_records, place_room_index)
from ouroboros.place_projection import circle_summary, connect_declared_place, place_view, places_view
from ouroboros.store import StoreError


def update_place(root, place_id: str, body: dict) -> dict:
    allowed = {"state", "wakes", "add_note", "remove_note", "working_folder", "model",
               "confirm_owner_account", "revoke_owner_account", "circle_id", "description", "expected_revision"}
    if not isinstance(body, dict) or not body or set(body) - allowed:
        raise PlaceError("place_update_invalid")
    actions = []
    mapping = {"state": ("set_state", "state"), "wakes": ("set_wakes", "wakes"),
               "add_note": ("add_note", "text"), "remove_note": ("remove_note", "note_id"),
               "working_folder": ("set_working_folder", "working_folder"), "model": ("set_model", "model"),
               "revoke_owner_account": ("revoke_owner_account", "platform_actor_id"),
               "circle_id": ("set_circle", "circle_id"), "description": ("set_description", "description")}
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
            raw = await request.body()
            body = await request.json() if raw else {}
            if not isinstance(body, dict) or set(body) - {"expected_revision"}:
                raise PlaceError("place_update_invalid")
            await run_sync_to_completion(edit_place, root, place_id, "disconnect", author=AUTHOR_OWNER,
                                         expected_revision=body.get("expected_revision"))
            return JSONResponse({"ok": True})
        body = await request.json()
        if place_id:
            return JSONResponse(await run_sync_to_completion(update_place, root, place_id, body))
        first_keys = {"note", "description", "circle_id"}  # what an Add says about the new place
        if not isinstance(body, dict) or set(body) - {"connector", "account_id", "conversation_id", "label", *first_keys}:
            raise PlaceError("place_connect_invalid")
        if not isinstance(body.get("connector"), str):
            raise PlaceError("place_connect_invalid", "connector")
        identity = {key: value for key, value in body.items() if key not in first_keys}
        first = {key: value for key, value in body.items() if key in first_keys}

        def connect():
            return {"ok": True, "place": place_view(root, connect_declared_place(root, **identity, first=first))}
        return JSONResponse(await run_sync_to_completion(connect))
    except (ValueError, TypeError) as exc:
        code = getattr(exc, "code", "")
        conflict = code in {"place_revision_conflict", "place_exists"}
        result = {"ok": False, "error": str(exc), "code": code or "place_invalid"}
        if conflict:
            conflict_place_id = place_id or getattr(exc, "place_id", "")
            def current():
                try:  # a repeated Add names the place connected already
                    return place_view(root, get_place(root, conflict_place_id))
                except PlaceError as error:
                    if error.code != "place_not_found":
                        raise
                    return None
            result["place"] = await asyncio.to_thread(current)
        return JSONResponse(result, status_code=409 if conflict else 400)


async def api_circles(request):
    """The owner's circle choices: name, words and model, one locked writer with the places."""
    root = request_drive_root(request)
    circle_id = str(request.path_params.get("circle_id") or "")
    try:
        raw = await request.body()
        body = await request.json() if raw else {}
        if not isinstance(body, dict):
            raise PlaceError("circle_update_invalid")
        if request.method == "DELETE":
            if set(body) - {"expected_revision"}:
                raise PlaceError("circle_update_invalid")
            removed = await run_sync_to_completion(delete_circle, root, circle_id,
                                                   expected_revision=body.get("expected_revision"))
            return JSONResponse({"ok": True, "circle": circle_summary(removed["circle"]), "detached": removed["detached"]})
        if circle_id:
            if not body or set(body) - {"name", "text", "model", "expected_revision"}:
                raise PlaceError("circle_update_invalid")
            saved = await run_sync_to_completion(edit_circle, root, circle_id, author=AUTHOR_OWNER, **body)
        else:
            if set(body) - {"name", "text", "model"}:
                raise PlaceError("circle_update_invalid")
            saved = await run_sync_to_completion(create_circle, root, name=body.get("name"), text=body.get("text", ""),
                                                 model=body.get("model", ""), author=AUTHOR_OWNER)
        return JSONResponse({"ok": True, "circle": circle_summary(saved)})
    except (ValueError, TypeError) as exc:
        conflict = getattr(exc, "code", "") == "circle_revision_conflict"
        result = {"ok": False, "error": str(exc), "code": getattr(exc, "code", "circle_invalid")}
        if conflict:
            def current():
                circle = place_records(root).circle(circle_id)
                return circle_summary(circle) if circle else None
            result["circle"] = await asyncio.to_thread(current)
        return JSONResponse(result, status_code=409 if conflict else 400)


async def api_place_rooms(request):
    """Lazy owner read of captured rooms, not remote-provider history or task admission."""
    try:
        place_id = str(request.path_params['place_id'])
        def read():
            from ouroboros import dialogue_store
            from ouroboros.places import get_place
            root = request_drive_root(request)
            get_place(root, place_id)  # An unknown Place never creates a dialogue store.
            coverage = dialogue_store.range_coverage(root, 'chat')
            return {'place_id': place_id, 'rooms': place_room_index(root, place_id, details=True),
                    'coverage': {'discovery': 'arrivals', 'importing': bool(coverage['pending']),
                                 'gaps': bool(coverage['mismatch'] or coverage['gaps'])}}
        return JSONResponse(await asyncio.to_thread(read), headers={'Cache-Control': 'no-store'})
    except PlaceError as exc:
        return JSONResponse({'ok': False, 'code': exc.code, 'error': str(exc)},
                            status_code=404 if exc.code == 'place_not_found' else 503)
    except (OSError, ValueError, StoreError) as exc:
        return JSONResponse({'ok': False, 'code': 'place_rooms_unavailable', 'error': str(exc)}, status_code=503)


def places_routes():
    return [Route("/api/places", api_places, methods=["GET", "POST"]),
            Route("/api/places/{place_id}/rooms", api_place_rooms, methods=["GET"]),
            Route("/api/places/{place_id}", api_places, methods=["POST", "DELETE"]),
            Route("/api/circles", api_circles, methods=["POST"]),
            Route("/api/circles/{circle_id}", api_circles, methods=["POST", "DELETE"])]
