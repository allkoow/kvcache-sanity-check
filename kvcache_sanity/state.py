from __future__ import annotations

import os
import tempfile
from pathlib import Path

from pydantic import BaseModel

from kvcache_sanity.models import ReferenceSet


def _write_atomic(path: Path, model: BaseModel) -> None:
    """Readers never see a partial file: a SIGKILL mid-write leaves the previous version."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            tmp_file.write(model.model_dump_json(indent=2))
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


class StateDir:
    """Artifacts phases hand over to each other, so each phase can run in a separate step."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    @property
    def reference_set_path(self) -> Path:
        return self.path / "reference_set.json"

    @property
    def control_dir(self) -> Path:
        return self.path / "control"

    @property
    def metrics_dir(self) -> Path:
        return self.path / "metrics"

    def save_reference_set(self, reference_set: ReferenceSet) -> None:
        _write_atomic(self.reference_set_path, reference_set)

    def load_reference_set(self) -> ReferenceSet:
        return ReferenceSet.model_validate_json(self.reference_set_path.read_text(encoding="utf-8"))
