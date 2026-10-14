"""Locate the files of the published LPAC release archive without hard-coded paths."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DATA_DIR_ENV_VAR = "LPAC_DATA_DIR"
RELEASE_URL = (
    "https://github.com/KumarRobotics/CoverageControl/releases/download/"
    "v1.3.1/lpac_CoverageControl.tar.gz"
)


@dataclass(frozen=True)
class DataPaths:
    """Paths inside the extracted ``lpac/`` folder of the release archive."""

    root: Path

    @property
    def model_state_dict(self) -> Path:
        return self.root / "models" / "model_k3_1024_state_dict.pt"

    @property
    def learning_params(self) -> Path:
        return self.root / "params" / "learning_params.toml"

    @property
    def env_params(self) -> Path:
        return self.root / "params" / "coverage_control_params.toml"

    @property
    def envs_dir(self) -> Path:
        return self.root / "envs"

    def env_file(self, index: int) -> Path:
        """Importance density field (Gaussian features) of environment ``index``."""
        return self.envs_dir / f"{index}.env"

    def pos_file(self, index: int) -> Path:
        """Initial robot positions of environment ``index`` (one ``x y`` pair per line)."""
        return self.envs_dir / f"{index}.pos"

    def count_environments(self) -> int:
        """Number of environments for which both ``.env`` and ``.pos`` files exist."""
        count = 0
        while self.env_file(count).is_file() and self.pos_file(count).is_file():
            count += 1
        return count

    def check(self) -> None:
        """Raise a clear error when a required file is missing."""
        for path in (self.model_state_dict, self.learning_params, self.env_params):
            if not path.is_file():
                raise FileNotFoundError(f"Missing release file: {path}")
        if self.count_environments() == 0:
            raise FileNotFoundError(f"No environment files found in {self.envs_dir}")


def resolve_data_dir(cli_value: str | None = None) -> DataPaths:
    """Resolve the data directory from the CLI, else from ``$LPAC_DATA_DIR``.

    The directory is the ``lpac/`` folder produced by extracting the release
    archive (it contains ``models/``, ``envs/`` and ``params/``).
    """
    value = cli_value or os.environ.get(DATA_DIR_ENV_VAR)
    if not value:
        raise SystemExit(
            f"Give --data-dir or set {DATA_DIR_ENV_VAR} to the extracted 'lpac' folder."
        )
    paths = DataPaths(Path(value).expanduser().resolve())
    paths.check()
    return paths
