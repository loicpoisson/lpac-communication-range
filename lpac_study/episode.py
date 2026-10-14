"""Run one controller for one episode and record costs and graph statistics."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import coverage_control as cc
from coverage_control import CoverageSystem
from coverage_control.algorithms import CentralizedCVT, ClairvoyantCVT, DecentralizedCVT

from .effective_rank import effective_rank
from .graph_stats import degree_stats
from .lpac_policy import LPACPolicy

CVT_ALGORITHMS = {
    "DecentralizedCVT": DecentralizedCVT,
    "CentralizedCVT": CentralizedCVT,
    "ClairvoyantCVT": ClairvoyantCVT,
}
LPAC_NAME = "lpac"
CONTROLLERS = (LPAC_NAME, *CVT_ALGORITHMS)


@dataclass
class EpisodeRecord:
    """Everything measured during one episode of one controller."""

    costs: np.ndarray  # cost / initial cost, index 0 is the initial state
    converged_step: int | None = None
    degree_mean: float = math.nan
    degree_std: float = math.nan
    isolated_fraction: float = math.nan
    layer_eranks: np.ndarray | None = None  # mean effective rank per GNN layer
    layer_eranks_std: np.ndarray | None = None
    _degrees: list = field(default_factory=list, repr=False)

    @property
    def cost_mean(self) -> float:
        """Average cost over the whole episode (the paper's 'average cost')."""
        return float(self.costs.mean())

    @property
    def cost_final(self) -> float:
        return float(self.costs[-1])


def _fill_after_convergence(costs: np.ndarray, step: int, value: float) -> None:
    """Freeze the cost curve once a controller stops moving, as the library evaluator does."""
    costs[step:] = value


def run_cvt_episode(
    algorithm_name: str, params: cc.Parameters, env: CoverageSystem, num_steps: int
) -> EpisodeRecord:
    """Run a CVT baseline (clairvoyant, centralized or decentralized)."""
    algorithm = CVT_ALGORITHMS[algorithm_name](params, env)
    initial_cost = env.GetObjectiveValue()
    costs = np.ones(num_steps)
    record = EpisodeRecord(costs=costs)
    for step in range(1, num_steps):
        algorithm.ComputeActions()
        if env.StepActions(algorithm.GetActions()):
            raise RuntimeError("Simulator reported an error while stepping")
        costs[step] = env.GetObjectiveValue() / initial_cost
        if algorithm.IsConverged():
            _fill_after_convergence(costs, step, costs[step])
            record.converged_step = step
            break
    return record


def run_lpac_episode(
    policy: LPACPolicy,
    params: cc.Parameters,
    env: CoverageSystem,
    num_steps: int,
    record_rank: bool = True,
) -> EpisodeRecord:
    """Run the LPAC policy and measure degree and effective rank at every step."""
    initial_cost = env.GetObjectiveValue()
    num_robots = env.GetNumRobots()
    costs = np.ones(num_steps)
    record = EpisodeRecord(costs=costs)
    degree_sum = degree_sq_sum = isolated_sum = 0.0
    eranks: list[list[float]] = []
    steps_run = 0
    for step in range(1, num_steps):
        data = policy.build_graph(env, params)
        stats = degree_stats(data.edge_index, num_robots)
        degree_sum += stats.mean
        degree_sq_sum += stats.std**2 + stats.mean**2
        isolated_sum += stats.isolated_fraction
        actions, converged, layer_outputs = policy.act(data)
        if record_rank:
            eranks.append([effective_rank(h) for h in layer_outputs])
        env.StepActions(actions)
        costs[step] = env.GetObjectiveValue() / initial_cost
        steps_run += 1
        if converged:
            _fill_after_convergence(costs, step, costs[step])
            record.converged_step = step
            break
    record.degree_mean = degree_sum / steps_run
    record.degree_std = math.sqrt(max(degree_sq_sum / steps_run - record.degree_mean**2, 0.0))
    record.isolated_fraction = isolated_sum / steps_run
    if record_rank and eranks:
        eranks_array = np.asarray(eranks)
        record.layer_eranks = eranks_array.mean(axis=0)
        record.layer_eranks_std = eranks_array.std(axis=0)
    return record
