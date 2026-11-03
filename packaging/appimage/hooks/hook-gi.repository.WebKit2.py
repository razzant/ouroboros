"""Collect WebKit2's typelib graph; helper/resources have a separate owner."""
from PyInstaller.utils.hooks.gi import GiModuleInfo

module_info = GiModuleInfo("WebKit2", "4.1")
if not module_info.available:
    raise RuntimeError("Linux desktop builds require WebKit2 4.1 introspection data")
binaries, datas, hiddenimports = module_info.collect_typelib_data()
