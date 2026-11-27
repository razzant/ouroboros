"""Synthetic store and canonical folder for registration custody tests."""
from ouroboros.gateways.claudexor import ClaudexorUnavailable

STORE = "fixture-store"
ROOT = "/fixture/project"


def binding(store=STORE, root=ROOT):
    return {"daemon_store_id": store, "registration_store_id": store, "registration_root": root}


def owner_facts(store=STORE, root=ROOT, **route):
    return {"daemon_store_id": store, **route, "project_resolution": {**binding(store, root), **route}}


class BoundRegistry:
    store_identity = STORE
    compares_candidates = True  # False: an engine that ignores ``candidate`` and answers without ``candidateMatch``

    def operations(self):
        return [{"method": "POST", "path": "/v2/projects/:id/release"},
                {"method": "GET", "path": "/v2/projects/:id/registration"}]

    def registered_root(self, project_id):
        return ROOT

    def folder(self, path):
        """This engine's own identity of a folder spelling: its alias table (``aliases``), never the host's disk."""
        return (getattr(self, "aliases", None) or {}).get(path, path)

    def bound_registration(self, project_id, store_id, root, *, release=False, candidate=""):
        if store_id != self.store_identity:
            raise ClaudexorUnavailable("store_mismatch", "different store", status_code=409)
        actual = self.registered_root(project_id)
        if actual is not None and actual != root:
            raise ClaudexorUnavailable("project_root_mismatch", "different folder", status_code=409)
        state = "absent" if actual is None else "present"
        if release:
            assert not candidate, "a release never carries a candidate"
            state = self.release_registered(project_id)
            if state not in ("released", "absent"):
                raise ClaudexorUnavailable("receipt_unverified", "unconfirmed release")
        reply = {"state": state, "projectId": project_id, "root": root, "storeId": store_id}
        if candidate:
            self.__dict__.setdefault("asked", []).append((project_id, candidate))
            if state == "present" and self.compares_candidates:
                reply["candidateMatch"] = "same" if self.folder(candidate) == self.folder(root) else "different"
        return reply

    def remove_project(self, project_id):
        raise AssertionError("automatic cleanup sent an unconditional DELETE")


class StartRegistry(BoundRegistry):
    """The register/find fixture API served through the paired protocol."""
    def project_registration_receipt(self, root):
        existing = self.find_project_id(root)
        self.canonical_root = root
        return {"id": existing or self.register_project(root), "created": not existing,
                "registration_root": root, "registration_store_id": self.store_identity}

    def registered_root(self, project_id):
        return self.canonical_root

    def release_registered(self, project_id):
        self.removals.append(project_id)
        self.canonical_root = None
        return "released"


def enable_start_registry(monkeypatch, gateway_class):
    """Equip a start fixture with bound registration while leaving its run API intact."""
    from ouroboros import delegate_registration_policy

    monkeypatch.setattr(delegate_registration_policy, "_CREATED_REPORTED", {})
    for name in ("store_identity", "operations", "registered_root", "bound_registration", "release_registered",
                 "project_registration_receipt", "remove_project"):
        monkeypatch.setattr(gateway_class, name, getattr(StartRegistry, name), raising=False)
