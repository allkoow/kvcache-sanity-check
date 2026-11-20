from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

from kvcache_sanity.models import ReferenceSet, ReplayRun

_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def write_atomic(path: Path, text: str) -> None:
    """Readers never see a partial file: a SIGKILL mid-write leaves the previous version."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            tmp_file.write(text)

        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


class StateDir:
    """Where a run saves its phase artifacts and keeps its checkpoint control files."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    @property
    def reference_set_path(self) -> Path:
        return self.path / "reference_set.json"

    @property
    def replays_dir(self) -> Path:
        return self.path / "replays"

    @property
    def control_dir(self) -> Path:
        return self.path / "control"

    def save_reference_set(self, reference_set: ReferenceSet) -> None:
        write_atomic(self.reference_set_path, reference_set.model_dump_json(indent=2))

    def replay_path(self, name: str) -> Path:
        if not _NAME_RE.fullmatch(name):
            raise ValueError(f"replay name {name!r} must match {_NAME_RE.pattern}")

        return self.replays_dir / f"{name}.json"

    def save_replay(self, replay: ReplayRun) -> None:
        write_atomic(self.replay_path(replay.name), replay.model_dump_json(indent=2))
