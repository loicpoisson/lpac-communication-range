"""Timed LPAC training with the authors' building blocks and a checkpoint per epoch.

    python train_launch.py learning_params.toml 1024 [--epochs N] [--seed 0] [--device auto]

This is the authors' ``python/training/train_lpac.py`` (CoverageControl v1.3.1)
with three additions and no change to the model or the data: fixed seeds, a
state-dict checkpoint and the loss curves written after every epoch, and the
wall time of every epoch logged to ``timing.csv``.  The epoch loop itself calls
``TrainModel.train_epoch`` / ``validate_epoch`` from the library unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import random
import sys
import time

import numpy as np
import torch
import torch_geometric
from coverage_control import IOUtils
from coverage_control.nn import LPAC, CNNGNNDataset, TrainModel

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # lpac_study, next to training/
from lpac_study.variants import (MultiScaleDataset, RangeDataset, build_model, parameter_count,  # noqa: E402
                                 receptive_field_hops, uses_comm_map, uses_multiscale_maps, uses_range_input)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda:0" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def build(config: dict, world_size: int, device: torch.device, num_workers: int, variant: str = "V0"):
    """Model, loaders and optimizer exactly as train_lpac.py builds them (variant V0 = the library model)."""
    dataset_path = pathlib.Path(IOUtils.sanitize_path(config["DataDir"]))
    data_dir = dataset_path / "data/"
    training = config["LPACTraining"]
    use_comm_map = config["ModelConfig"]["UseCommMaps"] and uses_comm_map(variant)
    model = build_model(variant, config).to(device)
    dataset_class = RangeDataset if uses_range_input(variant) else (MultiScaleDataset if uses_multiscale_maps(variant) else CNNGNNDataset)
    train_dataset = dataset_class(data_dir, "train", use_comm_map, world_size)
    val_dataset = dataset_class(data_dir, "val", use_comm_map, world_size)
    model.actions_mean = train_dataset.targets_mean.to(device)
    model.actions_std = train_dataset.targets_std.to(device)
    loader = torch_geometric.loader.DataLoader
    train_loader = loader(train_dataset, batch_size=training["BatchSize"], shuffle=True, num_workers=num_workers)
    val_loader = loader(val_dataset, batch_size=training["BatchSize"], shuffle=False, num_workers=num_workers)
    optimizer = torch.optim.Adam(model.parameters(), lr=training["LearningRate"], weight_decay=training["WeightDecay"])
    return model, train_loader, val_loader, optimizer, len(train_dataset), len(val_dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config_file")
    parser.add_argument("world_size", type=int)
    parser.add_argument("--epochs", type=int, default=None, help="override LPACTraining.NumEpochs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=None, help="override NumWorkers")
    parser.add_argument("--variant", default="V0", choices=("V0", "V1", "V2", "V3", "V0R", "V4", "V5"),
                        help="V0 original LPAC, V1 without neighbour map, V2 edge-geometry GNN, "
                             "V3 V1 + PointNetConv front layer, V0R V0 + own range as GNN input, "
                             "V4 V3 with bounded edge features, V5 V0 with multi-scale fixed maps (lpac_study.variants)")
    args = parser.parse_args()

    set_seeds(args.seed)
    config = IOUtils.load_toml(args.config_file)
    device = select_device(args.device)
    num_workers = config["NumWorkers"] if args.num_workers is None else args.num_workers
    num_epochs = config["LPACTraining"]["NumEpochs"] if args.epochs is None else args.epochs
    model_dir = pathlib.Path(IOUtils.sanitize_path(config["LPACModel"]["Dir"]))
    model_dir.mkdir(parents=True, exist_ok=True)

    tic = time.time()
    model, train_loader, val_loader, optimizer, n_train, n_val = build(config, args.world_size, device, num_workers, args.variant)
    load_time = time.time() - tic
    trainer = TrainModel(model, train_loader, val_loader, optimizer, torch.nn.MSELoss(), num_epochs, device, str(model_dir))
    print(f"device={device} variant={args.variant} train={n_train} val={n_val} epochs={num_epochs} workers={num_workers} "
          f"params={parameter_count(model)} hops={receptive_field_hops(args.variant, config)} load={load_time:.0f}s", flush=True)

    history = {"train": [], "val": []}
    best_val = float("inf")
    with (model_dir / "timing.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_s", "val_s", "elapsed_s"])
        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            train_loss = trainer.train_epoch()
            t1 = time.time()
            val_loss = trainer.validate_epoch(val_loader)
            t2 = time.time()
            history["train"].append(train_loss)
            history["val"].append(val_loss)
            torch.save(model.state_dict(), model_dir / f"model_epoch{epoch:03d}_state_dict.pt")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), model_dir / "model_best_state_dict.pt")
            (model_dir / "loss_history.json").write_text(json.dumps(history, indent=1))
            writer.writerow([epoch, train_loss, val_loss, round(t1 - t0, 1), round(t2 - t1, 1), round(t2 - tic, 1)])
            handle.flush()
            print(f"epoch {epoch}/{num_epochs} train {train_loss:.4e} val {val_loss:.4e} "
                  f"({t1 - t0:.0f}s + {t2 - t1:.0f}s)", flush=True)
    (model_dir / "run_info.json").write_text(json.dumps({
        "device": str(device), "variant": args.variant, "params": parameter_count(model),
        "receptive_field_hops": receptive_field_hops(args.variant, config),
        "epochs": num_epochs, "seed": args.seed, "num_workers": num_workers,
        "train_size": n_train, "val_size": n_val, "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "config_file": os.path.abspath(args.config_file), "total_s": round(time.time() - tic, 1),
    }, indent=1))


if __name__ == "__main__":
    main()
