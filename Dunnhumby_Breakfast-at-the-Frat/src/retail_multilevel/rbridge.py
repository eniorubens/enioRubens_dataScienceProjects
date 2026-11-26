"""Bridge to R (lme4 / glmmTMB) through rpy2, pinned to the R inside the conda env.

A machine-wide ``R_HOME`` (e.g. a system R in Program Files) would otherwise win, and
rpy2 would load an R without lme4 or fail to start. Always import rpy2 through
:func:`get_robjects` (or call :func:`configure_r_home` first).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def env_r_home() -> Path | None:
    """Return the R home of the active conda env, or ``None`` if the env has no R."""
    for candidate in (Path(sys.prefix) / "Lib" / "R", Path(sys.prefix) / "lib" / "R"):
        if (candidate / "bin").is_dir():
            return candidate
    return None


def configure_r_home() -> Path:
    """Point ``R_HOME`` (and ``PATH``, on Windows) to the env's R. Idempotent."""
    r_home = env_r_home()
    if r_home is None:
        raise RuntimeError(f"No R found in {sys.prefix}. Create the env from environment.yml.")
    os.environ["R_HOME"] = str(r_home)
    if sys.platform == "win32":
        # rpy2's API mode is unavailable on Windows; asking for ABI avoids a noisy fallback message.
        os.environ.setdefault("RPY2_CFFI_MODE", "ABI")
        bin_dir = str(r_home / "bin" / "x64")
        if bin_dir not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    return r_home


def get_robjects():
    """Configure R and return the ``rpy2.robjects`` module."""
    configure_r_home()
    import rpy2.robjects as ro

    return ro
