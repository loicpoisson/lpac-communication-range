"""Time the per-sample access of the V5 multi-scale dataset: whole-tensor COO indexing against the
per-sample split of ``MultiScaleDataset`` (diagnosis of the 30 min epochs of 19-20 October).

    python training/time_loader.py /path/mixedM train --samples 50
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lpac_study.variants import MultiScaleDataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("stage", nargs="?", default="train")
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()
    tic = time.time()
    comm = torch.load(f"{args.root}/data/{args.stage}/comm_maps.pt", weights_only=True).coalesce()
    print(f"sparse maps: shape {tuple(comm.shape)}, nnz {comm._nnz():,}, loaded in {time.time() - tic:.0f} s")
    gen = torch.Generator().manual_seed(0)
    idx = torch.randint(comm.shape[0], (args.samples,), generator=gen).tolist()
    tic = time.time()
    for i in idx:
        comm[i].to_dense()
    old = (time.time() - tic) / args.samples
    print(f"whole-tensor COO indexing: {old * 1000:.0f} ms per sample")
    tic = time.time()
    ds = MultiScaleDataset(f"{args.root}/data", args.stage, True, 1024)
    print(f"MultiScaleDataset (per-sample split) built in {time.time() - tic:.0f} s")
    tic = time.time()
    for i in idx:
        ds.comm_map(i)
    new = (time.time() - tic) / args.samples
    print(f"per-sample split: {new * 1000:.1f} ms per sample")
    same = all(torch.equal(comm[i].to_dense(), ds.comm_map(i)) for i in idx[:10])
    print(f"identical maps on 10 samples: {same}")
    n = comm.shape[0]
    print(f"estimated loader time per epoch of {n:,} samples, single process: old {old * n / 60:.0f} min, new {new * n / 60:.1f} min")


if __name__ == "__main__":
    main()
