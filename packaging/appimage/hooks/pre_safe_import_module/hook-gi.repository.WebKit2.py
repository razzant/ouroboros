"""GI creates this module at runtime; let Analysis run its collection hook."""


def pre_safe_import_module(api):
    api.add_runtime_module(api.module_name)
