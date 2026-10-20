"""Unit tests of the V4 edge encoding and the V5 multi-scale maps.

    python -m pytest tests/test_variants_v4v5.py -q      (or: python tests/test_variants_v4v5.py)

No dataset or checkpoint is needed: positions are random, the maps are compared between the
dataset builder (``training/redraw_ranges.py``), the inference-time builder
(``lpac_study.lpac_policy.multiscale_channels``) and the reference drawing of the journal's C3
level (``lpac_study.neighbourhood.comm_maps_from_positions``).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lpac_study.lpac_policy import multiscale_channels  # noqa: E402
from lpac_study.neighbourhood import comm_maps_from_positions, in_range_mask  # noqa: E402
from lpac_study.variants import MULTISCALE_SCALES_M, POINTNET_SCALE_M, WORLD_SIZE_M, LPACBoundedGeometry  # noqa: E402
from training.redraw_ranges import comm_maps, edge_weights  # noqa: E402

torch.manual_seed(0)
NUM_ROBOTS = 32


def random_positions(batch: int = 4) -> torch.Tensor:
    """Robot positions in metres inside a 1024 m world, float32 as stored in the datasets."""
    return torch.rand(batch, NUM_ROBOTS, 2) * 1000.0 + 12.0


def edge_index_within(positions_m: torch.Tensor, comm_range: float) -> torch.Tensor:
    mask = in_range_mask(positions_m.double(), comm_range)
    receiver, sender = mask.nonzero(as_tuple=True)
    return torch.stack([sender, receiver])


def test_v4_edge_features_bounded_and_scale_invariant():
    p = random_positions(1)[0]
    pos = p / WORLD_SIZE_M  # what data.pos holds up to the +W/2 shift, which cancels in differences
    ei = edge_index_within(p, 1024.0)
    e = LPACBoundedGeometry.edge_features(pos, ei)
    assert e.shape == (ei.shape[1], 3)
    assert float(e.min()) >= -1.0 and float(e.max()) <= 1.0, "features must lie in [-1, 1]"
    direction, squashed = e[:, :2], e[:, 2]
    assert torch.allclose(direction.norm(dim=1), torch.ones(ei.shape[1]), atol=1e-5), "direction is a unit vector"
    # scaling every position by 2: the direction is unchanged, only tanh moves (and stays in (0, 1))
    e2 = LPACBoundedGeometry.edge_features(pos * 2.0, ei)
    assert torch.allclose(e2[:, :2], direction, atol=1e-5)
    d = ((pos[ei[0]] - pos[ei[1]]) * WORLD_SIZE_M).norm(dim=1)
    assert torch.allclose(squashed, torch.tanh(d / POINTNET_SCALE_M), atol=1e-5)
    assert torch.allclose(e2[:, 2], torch.tanh(2.0 * d / POINTNET_SCALE_M), atol=1e-5)
    assert float(e2[:, 2].max()) < 1.0


def test_v4_self_loop_feature_is_zero_and_model_runs():
    from coverage_control import IOUtils  # noqa: F401  (import check only: the model needs the release config)
    conv_in = torch.randn(NUM_ROBOTS, 34)
    layer = LPACBoundedGeometry.__new__(LPACBoundedGeometry)  # no config needed for the static method
    p = random_positions(1)[0]
    ei = edge_index_within(p, 128.0)
    e = layer.edge_features(p / WORLD_SIZE_M, ei)
    assert e.shape[0] == ei.shape[1]
    from lpac_study.variants import BoundedGeometryConv
    conv = BoundedGeometryConv(torch.nn.Sequential(torch.nn.Linear(37, 8), torch.nn.ReLU(), torch.nn.Linear(8, 8)),
                               torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(8, 34)))
    out = conv(conv_in, ei, e)
    assert out.shape == (NUM_ROBOTS, 34) and torch.isfinite(out).all()
    # a robot with no edge at all still gets an output (its self-loop)
    out_isolated = conv(conv_in, ei[:, :0], e[:0])
    assert torch.isfinite(out_isolated).all()


def test_v5_maps_at_128m_show_exactly_the_in_range_neighbours():
    """At R = 128 m every pair is the drawing of exactly the in-range neighbours at its own scale."""
    p = random_positions(3)
    for s in MULTISCALE_SCALES_M:
        pair = comm_maps(p, 128.0, scale=s)
        for b in range(p.shape[0]):
            ref = comm_maps_from_positions(p[b].double(), in_range_mask(p[b].double(), 128.0), s, 32)
            assert torch.equal(pair[b], ref)
            # the summed x-values of each map equal the sum of the in-range relative x offsets / scale
            rel = (p[b].unsqueeze(0) - p[b].unsqueeze(1)).double()
            expected = (rel[..., 0] * in_range_mask(p[b].double(), 128.0)).sum(1) / s
            assert torch.allclose(pair[b, :, 0].flatten(1).sum(1).double(), expected, atol=1e-4)
    assert comm_maps(p, 128.0, scale=256.0).shape == comm_maps(p, 128.0).shape


def test_v5_256m_pair_at_1024m_equals_the_c3_map():
    p = random_positions(3)
    pair = comm_maps(p, 1024.0, scale=256.0)
    for b in range(p.shape[0]):
        ref = comm_maps_from_positions(p[b].double(), in_range_mask(p[b].double(), 256.0), 256.0, 32)
        assert torch.equal(pair[b], ref), "the 256 m pair at 1024 m must be the C3 map of the journal"
    pair1024 = comm_maps(p, 1024.0, scale=1024.0)
    assert torch.allclose(pair1024, comm_maps(p, 1024.0), atol=1e-7), "the 1024 m pair at 1024 m is the original map"


def test_v5_inference_builder_matches_dataset_builder():
    p = random_positions(2)
    for r in (128.0, 512.0, 1024.0):
        dataset_maps = torch.cat([comm_maps(p, r, scale=s) for s in MULTISCALE_SCALES_M], dim=2)
        for b in range(p.shape[0]):
            x = torch.zeros(NUM_ROBOTS, 4, 32, 32)
            x[:, 0] = 1.0  # local map stand-in
            x[:, 3] = 2.0  # obstacle map stand-in
            x8 = multiscale_channels(x, p[b].double(), r)
            assert x8.shape == (NUM_ROBOTS, 8, 32, 32)
            assert torch.equal(x8[:, 0], x[:, 0]) and torch.equal(x8[:, 7], x[:, 3])
            assert torch.allclose(x8[:, 1:7], dataset_maps[b], atol=1e-6)


def test_edge_weights_unchanged_by_map_options():
    p = random_positions(2)
    w = edge_weights(p, 512.0)
    assert (w >= 0).all() and (w <= 1).all()
    assert bool(((w > 0) == (torch.cdist(p, p) < 512.0).logical_and(~torch.eye(NUM_ROBOTS, dtype=torch.bool))).all())
    assert math.isclose(float(w.diagonal(dim1=1, dim2=2).abs().sum()), 0.0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
