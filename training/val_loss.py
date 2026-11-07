"""Validation loss of a stored checkpoint, with no training: does a rebuilt dataset match the old one?

    python training/val_loss.py learning_params.toml 1024 --checkpoint model_epoch035_state_dict.pt

Loads the validation split that ``DataDir`` points at and runs the library's ``validate_epoch``
on it, with the same loss (MSE on normalized actions), batch size and normalization as
``train_launch.py``. The 256 m dataset was rebuilt from the 128 m positions after the original
was deleted, so its samples are not the ones the stored model was trained on; this reports the
gap, which has to stay within the epoch-to-epoch swing of the original run for replicates on the
rebuilt data to be comparable with it.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import torch
from coverage_control.nn import TrainModel

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from training.train_launch import build, select_device, set_seeds  # noqa: E402

from coverage_control import IOUtils  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config_file")
    parser.add_argument("world_size", type=int)
    parser.add_argument("--checkpoint", required=True, help="state dict to score")
    parser.add_argument("--variant", default="V0")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    set_seeds(args.seed)
    device = select_device(args.device)
    config = IOUtils.load_toml(args.config_file)
    model, train_loader, val_loader, optimizer, n_train, n_val = build(config, args.world_size, device,
                                                                       args.num_workers, args.variant)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"state dict: {len(missing)} missing, {len(unexpected)} unexpected keys")
    trainer = TrainModel(model, train_loader, val_loader, optimizer, torch.nn.MSELoss(), 0, device, ".")
    loss = trainer.validate_epoch(val_loader)
    print(f"checkpoint {pathlib.Path(args.checkpoint).name}")
    print(f"data       {config['DataDir']} ({n_val} validation samples, {n_train} training samples)")
    print(f"val_loss   {loss:.6f}")


if __name__ == "__main__":
    main()
