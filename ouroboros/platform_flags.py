"""The platform IDENTITY constants, and nothing else.

Its own module so the two platform-layer files can both read them without either
importing the other. Deliberately tiny and dependency-free: the moment this grows a
behaviour it stops being a constant and belongs in the layer proper.
"""

from __future__ import annotations

import sys

IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")

# What a boot identity reads as where there is no boot id to read. The value says WHY
# rather than merely that it is absent, and it is compared and serialized, so it is the
# string itself that is the contract — not the name.
BOOT_IDENTITY_UNKNOWN = "non-linux"
