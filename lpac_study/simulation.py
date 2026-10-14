"""Thin wrappers around the CoverageControl simulator shared by all experiments.

Everything here is deterministic: the simulator only draws random numbers when
it generates a new environment or when position noise is enabled, and both
are avoided (environments come from the release archive, noise is off).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import coverage_control as cc
from coverage_control import CoverageSystem, PointVector, WorldIDF


def make_parameters(
    env_params_file: Path, communication_range: float, num_robots: int
) -> cc.Parameters:
    """Load the authors' simulation parameters and override range and team size.

    ``CoverageSystem`` copies the parameters at construction, so the overrides
    must happen before the environment is built.
    """
    params = cc.Parameters(str(env_params_file))
    params.pCommunicationRange = float(communication_range)
    params.pNumRobots = int(num_robots)
    return params


def load_initial_positions(pos_file: Path, num_robots: int) -> np.ndarray:
    """Return the first ``num_robots`` initial positions stored in a ``.pos`` file.

    The released files hold 32 positions.  The simulator refuses a ``.pos``
    file whose length differs from ``NumRobots``, so smaller teams are built by
    keeping the first rows of the file (a fixed, documented, seed-free choice).
    """
    positions = np.loadtxt(pos_file, dtype=np.float64).reshape(-1, 2)
    if num_robots > len(positions):
        raise ValueError(
            f"{pos_file} holds {len(positions)} positions, fewer than {num_robots}"
        )
    return positions[:num_robots]


def make_environment(
    params: cc.Parameters, env_file: Path, initial_positions: np.ndarray
) -> CoverageSystem:
    """Build a fresh coverage environment from an IDF file and initial positions."""
    world_idf = WorldIDF(params, str(env_file))
    return CoverageSystem(params, world_idf, PointVector(initial_positions))


def normalized_cost(env: CoverageSystem, initial_cost: float) -> float:
    """Coverage cost divided by the cost at the start of the episode (as in the paper)."""
    return env.GetObjectiveValue() / initial_cost
