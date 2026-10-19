"""Linear CKA between the layer embeddings of two LPAC specialists on the same states (control of H1).

    python -m lpac_study.cka --baseline results/results.csv \
        --pairs "r128 s0 vs r1024 s0=/path/r128_s0.pt,/path/r1024.pt" "r128 s0 vs r128 s1=/path/r128_s0.pt,/path/r128_s1.pt" \
        --env 0 --fraction 0.5 --steps 0 50 100 150 200 250 300

States come from one heterogeneous episode (fraction ``f`` of robots at 1024 m, draw 0, as-is maps)
driven by the published model; at each listed step both models of a pair are run on the same
inputs and the CKA (Kornblith et al., 2019, linear kernel, centred) is computed between their
CNN outputs, each TAGConv layer output and their actions, over the robots of all listed states.
CKA = 1 means the two representations are the same up to a linear map; the question is whether
two models trained from the same seed (same initial weights and sample order) are closer than
two models trained from different seeds.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from coverage_control import IOUtils

from .factorial import NUM_ROBOTS
from .heterogeneous import HeterogeneousPolicy, long_range_robots
from .paths import resolve_data_dir
from .simulation import load_initial_positions, make_environment, make_parameters
from .variants import build_model


def linear_cka(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean(0, keepdim=True)
    b = b - b.mean(0, keepdim=True)
    num = (a.T @ b).norm() ** 2
    den = (a.T @ a).norm() * (b.T @ b).norm()
    return float(num / den) if den > 0 else float("nan")


def layer_outputs(model, data) -> dict[str, torch.Tensor]:
    """CNN output, each TAGConv layer output (after ReLU), MLP output and de-normalized actions."""
    x, edge_index, pos = data.x, data.edge_index, data.pos
    out = {}
    h = model.cnn_backbone(x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1]))
    out["cnn"] = h
    h = torch.cat([h, pos], dim=-1)
    for i in range(model.gnn_backbone.num_layers):
        h = torch.relu(model.gnn_backbone._modules[f"graph_conv_{i}"](h, edge_index))
        out[f"gnn{i + 1}"] = h
    h = model.gnn_mlp(h)
    out["mlp"] = h
    out["action"] = model.output_linear(h) * model.actions_std + model.actions_mean
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--pairs", nargs="+", required=True, help='"label=path_a,path_b" entries')
    parser.add_argument("--env", type=int, default=0)
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--draw", type=int, default=0)
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 50, 100, 150, 200, 250, 300])
    parser.add_argument("--short-range", type=float, default=128.0)
    parser.add_argument("--long-range", type=float, default=1024.0)
    args = parser.parse_args()
    paths = resolve_data_dir(args.data_dir)
    device = torch.device("cpu")
    params_cfg = IOUtils.load_toml(str(paths.learning_params))
    models = {}
    for item in args.pairs:
        for path in item.split("=", 1)[1].split(","):
            if path not in models:
                m = build_model("V0", params_cfg).to(device)
                m.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
                m.eval()
                models[path] = m
    # states from one heterogeneous episode driven by the published model
    is_long = np.zeros(NUM_ROBOTS, dtype=bool)
    is_long[long_range_robots(args.env, args.draw, args.fraction)] = True
    ranges = torch.full((NUM_ROBOTS,), args.short_range, dtype=torch.float64)
    ranges[is_long] = args.long_range
    driver = HeterogeneousPolicy(paths, device, ranges, "asis")
    params = make_parameters(paths.env_params, args.long_range, NUM_ROBOTS)
    env = make_environment(params, paths.env_file(args.env), load_initial_positions(paths.pos_file(args.env), NUM_ROBOTS))
    states = []
    with torch.no_grad():
        for step in range(max(args.steps) + 1):
            data = driver.build_graph(env, params)
            if step in args.steps:
                states.append(data.clone())
            actions, _, _ = driver.act(data)
            env.StepActions(actions)
    print(f"{len(states)} states from env {args.env}, f = {args.fraction}, draw {args.draw} ({int(is_long.sum())} long-range robots)")
    layers = None
    results = {}
    with torch.no_grad():
        for item in args.pairs:
            label, paths_ab = item.split("=", 1)
            pa, pb = paths_ab.split(",")
            outs_a = [layer_outputs(models[pa], d) for d in states]
            outs_b = [layer_outputs(models[pb], d) for d in states]
            layers = list(outs_a[0])
            row = {}
            for layer in layers:
                a = torch.cat([o[layer] for o in outs_a]); b = torch.cat([o[layer] for o in outs_b])
                row[layer] = linear_cka(a, b)
                # also per robot group
                mask = torch.tensor(np.tile(is_long, len(states)))
                row[layer + "/short"] = linear_cka(a[~mask], b[~mask])
                row[layer + "/long"] = linear_cka(a[mask], b[mask])
            results[label] = row
    print("linear CKA per layer (all robots; in brackets short-range robots / long-range robots):")
    header = "pair".ljust(26) + " ".join(l.rjust(16) for l in layers)
    print(header)
    for label, row in results.items():
        print(label.ljust(26) + " ".join(f"{row[l]:.2f} [{row[l + '/short']:.2f}/{row[l + '/long']:.2f}]".rjust(16) for l in layers))


if __name__ == "__main__":
    main()
