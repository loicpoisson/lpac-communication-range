"""The training graph of the heterogeneous dataset (item 2) is the evaluation graph.

    python tests/test_hetero_fraction_dataset.py

Random positions only, no dataset on disk. The point of the first test is the one that matters:
`training.redraw_ranges` builds the graph a model is trained on, `lpac_study.heterogeneous` builds
the graph it is evaluated on, and the two must be identical for the same positions, ranges and
link rule, or the comparison against V4 mixed means nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lpac_study.heterogeneous import SEMANTICS, graph_mask, reception_edge_index  # noqa: E402
from training.redraw_ranges import (HETERO_FRACTIONS, LONG_RANGE, SHORT_RANGE, batched_graph_mask,  # noqa: E402
                                    draw_fraction_ranges, fraction_edge_weights)

torch.manual_seed(0)
BATCH, NUM_ROBOTS = 8, 32


def positions() -> torch.Tensor:
    return torch.rand(BATCH, NUM_ROBOTS, 2, dtype=torch.float32) * 1000.0 + 12.0


def test_training_graph_equals_evaluation_graph():
    p = positions()
    ranges = draw_fraction_ranges(BATCH, NUM_ROBOTS, seed=3)
    for semantics in SEMANTICS:
        batched = batched_graph_mask(p, ranges, semantics)
        for b in range(BATCH):
            expected = graph_mask(p[b].double(), ranges[b].double(), semantics)
            assert torch.equal(batched[b], expected), f"{semantics}, sample {b}"


def test_stored_weights_carry_exactly_those_edges():
    """The file keeps W[j, i], so its sparse indices must be the evaluation edge_index."""
    p = positions()
    ranges = draw_fraction_ranges(BATCH, NUM_ROBOTS, seed=4)
    for semantics in SEMANTICS:
        weights = fraction_edge_weights(p, ranges, semantics)
        for b in range(BATCH):
            stored = torch.stack((weights[b] != 0).nonzero(as_tuple=True))  # rows = source, target
            expected = reception_edge_index(graph_mask(p[b].double(), ranges[b].double(), semantics))
            order_s = torch.argsort(stored[0] * NUM_ROBOTS + stored[1])
            order_e = torch.argsort(expected[0] * NUM_ROBOTS + expected[1])
            assert torch.equal(stored[:, order_s], expected[:, order_e]), f"{semantics}, sample {b}"


def test_weights_are_positive_on_links_and_zero_elsewhere():
    p = positions()
    ranges = draw_fraction_ranges(BATCH, NUM_ROBOTS, seed=5)
    weights = fraction_edge_weights(p, ranges, "transmission")
    mask = batched_graph_mask(p, ranges, "transmission").transpose(1, 2)
    assert bool((weights[mask] > 0).all()) and bool((weights[~mask] == 0).all())
    assert float(weights.max()) <= 1.0
    idx = torch.arange(NUM_ROBOTS)
    assert float(weights[:, idx, idx].abs().max()) == 0.0, "no self-loops"


def test_the_fraction_is_drawn_from_the_five_values_only():
    ranges = draw_fraction_ranges(4000, NUM_ROBOTS, seed=6)
    assert set(ranges.unique().tolist()) <= {SHORT_RANGE, LONG_RANGE}
    fractions = (ranges == LONG_RANGE).float().mean(dim=1)
    assert set(torch.round(fractions * 4).div(4).unique().tolist()) <= set(HETERO_FRACTIONS)
    for f in HETERO_FRACTIONS:  # every value appears, roughly one fifth of the samples each
        share = float((torch.isclose(fractions, torch.tensor(f))).float().mean())
        assert 0.15 < share < 0.25, f"fraction {f} drawn {share:.3f} of the time"


def test_the_draw_is_reproducible_and_seed_dependent():
    a = draw_fraction_ranges(64, NUM_ROBOTS, seed=1)
    assert torch.equal(a, draw_fraction_ranges(64, NUM_ROBOTS, seed=1))
    assert not torch.equal(a, draw_fraction_ranges(64, NUM_ROBOTS, seed=2))


def test_transmission_gives_short_robots_more_neighbours_than_reception():
    p = positions()
    ranges = draw_fraction_ranges(BATCH, NUM_ROBOTS, seed=7)
    short = ranges == SHORT_RANGE
    rec = batched_graph_mask(p, ranges, "reception")
    tra = batched_graph_mask(p, ranges, "transmission")
    assert int(tra[short].sum()) > int(rec[short].sum())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
