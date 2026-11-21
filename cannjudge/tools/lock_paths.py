"""Lock paths shared by standalone tools and a colocated ACW installation."""

import os
from pathlib import Path


def runtime_dir():
    acw = Path("/workspace/acw/var/run")
    if acw.is_dir():
        return acw
    return Path(os.environ.get("XDG_RUNTIME_DIR") or "/tmp") / f"cannjudge-{os.getuid()}"
