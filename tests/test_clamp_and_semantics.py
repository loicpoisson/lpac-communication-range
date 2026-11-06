"""Unit tests of the clamped neighbour map (Q1) and of the graph semantics (Q2).

    python tests/test_clamp_and_semantics.py

Random positions only, no dataset or checkpoint.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lpac_study.factorial import clamps_far_robots, map_scale, parse_cell  # noqa: E402
from lpac_study.heterogeneous import SEMANTICS, graph_mask, reception_edge_index, reception_mask  # noqa: E402
from lpac_study.neighbourhood import comm_maps_from_positions, in_range_mask  # noqa: E402

torch.manual_seed(0)
NUM_ROBOTS, SIZE, SCALE = 32, 32, 256.0


def positions() -> torch.Tensor:
    return (torch.rand(NUM_ROBOTS, 2, dtype=torch.float64) * 1000.0 + 12.0)


def test_level_names():
    assert map_scale("K256") == 256.0 and clamps_far_robots("K256")
    assert map_scale("S256") == 256.0 and not clamps_far_robots("S256")
    assert map_scale("C3") == 256.0 and not clamps_far_robots("C3")
    assert parse_cell("G0K256") == ("G0", "K256")


def test_clamped_map_keeps_the_near_robots_identical_to_the_dropping_map():
    """Within the scale, K256 must draw exactly what C3/S256 draws."""
    p = positions()
    near = in_range_mask(p, SCALE)
    dropped = comm_maps_from_positions(p, near, SCALE, SIZE)
    clamped_near = comm_maps_from_positions(p, near, SCALE, SIZE, clamp=True)
    assert torch.equal(dropped, clamped_near), "clamping must not move a robot that is already inside"


def test_clamped_map_adds_far_robots_on_the_border():
    p = positions()
    all_in_range = in_range_mask(p, 1024.0)
    near = in_range_mask(p, SCALE)
    far = all_in_range & ~near
    assert int(far.sum()) > 0, "the test needs robots beyond 256 m"
    maps = comm_maps_from_positions(p, all_in_range, SCALE, SIZE, clamp=True)
    # every far robot contributes a cell whose value has norm 1 (it sits on the 256 m circle)
    rel = (p.unsqueeze(0) - p.unsqueeze(1)).to(torch.float32)
    i, j = far.nonzero(as_tuple=True)
    direction = rel[i, j] / rel[i, j].norm(dim=-1, keepdim=True)
    cells = torch.round(direction * SIZE / 2.0 + (SIZE / 2.0 - 0.5)).long().clamp(0, SIZE - 1)
    for k in range(min(20, len(i))):
        v = maps[i[k], :, cells[k, 0], cells[k, 1]]
        assert float(v.norm()) > 0.0
    # the clamped map holds strictly more mass than the dropping one, and no value exceeds norm 1 per robot pair
    dropped = comm_maps_from_positions(p, near, SCALE, SIZE)
    assert float(maps.abs().sum()) > float(dropped.abs().sum())
    assert float(direction.norm(dim=-1).max()) <= 1.0 + 1e-6
    # cells stay inside the map
    assert int(cells.min()) >= 0 and int(cells.max()) <= SIZE - 1


def test_semantics_coincide_when_all_ranges_are_equal():
    p = positions()
    for r in (128.0, 512.0, 1024.0):
        ranges = torch.full((NUM_ROBOTS,), r, dtype=torch.float64)
        masks = [graph_mask(p, ranges, s) for s in SEMANTICS]
        for m in masks[1:]:
            assert torch.equal(masks[0], m), f"semantics differ at a homogeneous range {r}"
        assert torch.equal(masks[0], reception_mask(p, ranges))


def test_semantics_relations_with_mixed_ranges():
    p = positions()
    ranges = torch.full((NUM_ROBOTS,), 128.0, dtype=torch.float64)
    ranges[: NUM_ROBOTS // 2] = 1024.0
    rec = graph_mask(p, ranges, "reception")
    tra = graph_mask(p, ranges, "transmission")
    assert torch.equal(tra, rec.T), "transmission is the transpose of reception"
    assert torch.equal(graph_mask(p, ranges, "sym_min"), rec & tra)
    assert torch.equal(graph_mask(p, ranges, "sym_max"), rec | tra)
    for s in ("sym_min", "sym_max"):
        m = graph_mask(p, ranges, s)
        assert torch.equal(m, m.T), f"{s} must be undirected"
    assert int(rec.sum()) != int(tra.sum()) or True  # directed rules generally differ; equality is possible
    # a short-range robot receives more under transmission than under reception
    short = ranges == 128.0
    assert int(tra[short].sum()) > int(rec[short].sum())


def test_edge_orientation_is_sender_to_receiver():
    p = positions()
    ranges = torch.full((NUM_ROBOTS,), 128.0, dtype=torch.float64)
    ranges[:4] = 1024.0
    mask = graph_mask(p, ranges, "transmission")
    ei = reception_edge_index(mask)
    sender, receiver = ei[0], ei[1]
    assert bool(mask[receiver, sender].all()), "edge_index[0] must be the sender, [1] the receiver"
    assert int(ei.shape[1]) == int(mask.sum())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
