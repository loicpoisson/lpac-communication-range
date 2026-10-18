"""Mixture of range specialists at inference in a heterogeneous team (experiment H1).

    python -m lpac_study.mixture --baseline results/results.csv --out-dir results_h1 \
        --specialists 128=/path/r128/model_best_state_dict.pt 1024=/path/r1024/model_best_state_dict.pt \
        --fractions 0 0.25 0.5 0.75 1 --draws 0 1 --num-envs 11

Each robot runs the LPAC-K3 model trained at its own communication range (``--specialists``),
in a team where a fraction ``f`` of the robots has the long range (same draws, reception
semantics and map modes as ``lpac_study.heterogeneous``). The two models share the
architecture, so the mixture is computed layer by layer on the mixed inputs: the CNN of each
model on every robot's maps, keeping for each robot the output of its own model; then, for
each of the five TAGConv layers, both models' layers on the same hidden state and the same
graph, again keeping each robot's own model; the action MLP and the de-normalization likewise.
The messages that robot ``i`` aggregates therefore come from embeddings produced by its
neighbours' models, not by its own: the hypothesis under test is that two specialists do not
speak the same "message language", so that the mixture can be worse than either specialist
used by the whole team.

``--mode`` selects what is compared: ``mixture`` (each robot its specialist), ``short`` (the
short-range specialist on every robot), ``long`` (the long-range specialist on every robot).
The heterogeneous inputs are the same in the three modes; ``f = 0`` and ``f = 1`` are the
specialist-alone references at the homogeneous ranges.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from coverage_control import IOUtils, PointVector

from .factorial import NUM_ROBOTS, load_clairvoyant
from .heterogeneous import FIELDS as HETERO_FIELDS, HeterogeneousPolicy, long_range_robots, run_episode, subset_features
from .paths import resolve_data_dir
from .simulation import load_initial_positions, make_environment, make_parameters
from .variants import build_model

FIELDS = HETERO_FIELDS[:5] + ["mode"] + HETERO_FIELDS[5:]


class MixturePolicy(HeterogeneousPolicy):
    """Heterogeneous inputs, and one LPAC model per robot chosen by the robot's range."""

    def __init__(self, paths, device, ranges: torch.Tensor, map_mode: str, specialists: dict[float, str], mode: str):
        super().__init__(paths, device, ranges, map_mode)  # the published model, used only to build inputs
        params = IOUtils.load_toml(str(paths.learning_params))
        self.models = {}
        for r, path in specialists.items():
            model = build_model("V0", params).to(device)
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
            model.eval()
            self.models[float(r)] = model
        if mode == "mixture":
            assignment = self.ranges
        elif mode == "short":
            assignment = torch.full_like(self.ranges, min(self.models))
        elif mode == "long":
            assignment = torch.full_like(self.ranges, max(self.models))
        else:
            raise ValueError("mode must be mixture, short or long")
        self.keys = sorted(self.models)
        self.node_model = torch.tensor([self.keys.index(float(r)) for r in assignment.tolist()], device=device)
        self.num_layers = self.models[self.keys[0]].gnn_backbone.num_layers

    def _select(self, outputs: list[torch.Tensor]) -> torch.Tensor:
        """Row ``i`` of the output of the model assigned to robot ``i``."""
        stacked = torch.stack(outputs)  # (models, N, ...)
        index = self.node_model.view(1, -1, *([1] * (stacked.dim() - 2))).expand(1, *stacked.shape[1:])
        return stacked.gather(0, index).squeeze(0)

    @torch.no_grad()
    def act(self, data):
        x, edge_index, pos = data.x, data.edge_index, data.pos
        images = x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1])
        models = [self.models[k] for k in self.keys]
        h = self._select([m.cnn_backbone(images) for m in models])
        h = torch.cat([h, pos], dim=-1)
        for layer in range(self.num_layers):
            h = self._select([torch.relu(m.gnn_backbone._modules[f"graph_conv_{layer}"](h, edge_index)) for m in models])
        out = self._select([m.output_linear(m.gnn_mlp(h)) for m in models])
        std = self._select([m.actions_std.expand(out.shape[0], -1) for m in models])
        mean = self._select([m.actions_mean.expand(out.shape[0], -1) for m in models])
        actions = out * std + mean
        converged = bool(torch.allclose(actions, torch.zeros_like(actions), atol=1e-5))
        return PointVector(actions.cpu().numpy()), converged, []


def done_keys(path: Path) -> set[tuple]:
    if not path.is_file():
        return set()
    with path.open(newline="") as handle:
        return {(int(r["env"]), float(r["fraction"]), int(r["draw"]), r["map"], r["mode"]) for r in csv.DictReader(handle)}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--out-dir", default="results_h1")
    parser.add_argument("--specialists", nargs=2, required=True, help="two range=state_dict pairs (short and long range)")
    parser.add_argument("--modes", nargs="+", default=["mixture", "short", "long"], choices=["mixture", "short", "long"])
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--draws", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--maps", nargs="+", default=["asis"], choices=["asis", "fixed256"])
    parser.add_argument("--num-envs", type=int, default=11)
    parser.add_argument("--env-start", type=int, default=0)
    parser.add_argument("--env-end", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=600)
    args = parser.parse_args(argv)

    specialists = {float(item.split("=", 1)[0]): item.split("=", 1)[1] for item in args.specialists}
    short_range, long_range = min(specialists), max(specialists)
    paths = resolve_data_dir(args.data_dir)
    device = torch.device("cpu")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = out / "results_h1.csv"
    done = done_keys(results)
    clairvoyant = load_clairvoyant(Path(args.baseline), NUM_ROBOTS)
    env_end = args.num_envs if args.env_end is None else min(args.env_end, args.num_envs)
    started = time.time()
    for env_index in range(args.env_start, env_end):
        for fraction in args.fractions:
            for draw in args.draws:
                if fraction in (0.0, 1.0) and draw != args.draws[0]:
                    continue
                for map_mode in args.maps:
                    for mode in args.modes:
                        if fraction in (0.0, 1.0) and mode == "mixture":
                            continue  # identical to one specialist alone: measured under that mode
                        key = (env_index, float(fraction), draw, map_mode, mode)
                        if key in done:
                            continue
                        long_idx = long_range_robots(env_index, draw, fraction)
                        is_long = np.zeros(NUM_ROBOTS, dtype=bool)
                        is_long[long_idx] = True
                        ranges = torch.full((NUM_ROBOTS,), short_range, dtype=torch.float64)
                        ranges[is_long] = long_range
                        policy = MixturePolicy(paths, device, ranges, map_mode, specialists, mode)
                        params = make_parameters(paths.env_params, long_range, NUM_ROBOTS)
                        positions = load_initial_positions(paths.pos_file(env_index), NUM_ROBOTS)
                        env = make_environment(params, paths.env_file(env_index), positions)
                        clair_mean, clair_final = clairvoyant[env_index]
                        tic = time.time()
                        features = subset_features(env, positions, is_long, params.pWorldMapSize)
                        rec = run_episode(policy, None, params, env, args.num_steps, 0, is_long)
                        rec.update(features)
                        row = {"env": env_index, "fraction": fraction, "draw": draw, "map": map_mode, "mode": mode,
                               "model": f"{short_range:.0f}+{long_range:.0f}", "num_long": int(is_long.sum()),
                               "long_range": long_range, "short_range": short_range,
                               "ratio_mean": rec["cost_mean"] / clair_mean, "ratio_final": rec["cost_final"] / clair_final,
                               "clairvoyant_cost_mean": clair_mean, "wall_s": round(time.time() - tic, 1), **rec}
                        new = not results.is_file()
                        with results.open("a", newline="") as handle:
                            writer = csv.DictWriter(handle, fieldnames=FIELDS)
                            if new:
                                writer.writeheader()
                            writer.writerow({k: row.get(k) for k in FIELDS})
                        done.add(key)
                        print(f"[{time.time() - started:6.0f}s] env={env_index:2d} f={fraction:.2f} draw={draw} {map_mode:8s} "
                              f"{mode:7s} ratio={row['ratio_mean']:.3f} deg={row['degree_mean']:5.2f} ({row['wall_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
