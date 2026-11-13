import os
from pathlib import Path

APP_NAME = "kvcache-sanity-check"


def user_data_dir() -> Path:
    """Directory for user-supplied corpus documents and scenario files.

    Merged on top of the bundled corpus/scenarios (user entries win on a
    matching doc_id/scenario id). Override with KVCACHE_DATA_DIR, or place
    it under XDG_DATA_HOME; otherwise defaults to ~/.local/share/<app>,
    following the XDG Base Directory spec.
    """
    override = os.environ.get("KVCACHE_DATA_DIR")
    if override:
        return Path(override)
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg_data_home) if xdg_data_home else Path.home() / ".local" / "share"
    return base / APP_NAME
