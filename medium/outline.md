# Medium post: detailed outline

Final deliverable of the CS224W project (due 10 December 2026): a Medium post, a Colab notebook,
and the Medium account name. This file is the plan of the post; the draft will live next to it in
`medium/draft.md`.

**Reader.** A CS224W student or an ML engineer who knows what a GNN is but has never heard of
coverage control. Every robotics term is defined on first use; every claim points to a figure or
to the Colab.

**Length.** About 3,500 words, a 14-16 minute read, 6 figures and one table. Equations are
images or Medium code blocks, since Medium has no LaTeX.

**Working title.** *When more communication hurts: debugging a graph neural network for robot
swarms.* Alternative: *Why a GNN-controlled swarm gets worse when its robots can talk farther.*

**One-sentence story.** A published GNN controller for robot swarms gets 3.6 times worse when the
robots' communication range grows from 256 m to 1024 m. We trace the failure to one input image,
not to the graph, fix it at inference with no retraining, and design a model whose cost keeps
falling as the range grows.

---

## 1. Hook (about 250 words)

- One animation: the published controller at 1024 m next to the same controller with the fixed
  map (site replays, exported as a GIF).
- The paradox in one line: more communication should never hurt, yet the cost goes from 1.70 at
  256 m to 6.15 at 1024 m (40 new environments; cost divided by the clairvoyant optimum, lower is
  better).
- What the reader will learn: how to diagnose a GNN by intervening on its inputs one at a time.

## 2. The problem and the model (about 600 words)

- **Coverage control** in one paragraph: 32 robots spread over a 1024 m square to sit where an
  unknown importance density is high. The cost and the clairvoyant CVT baseline, with one figure
  of a density and the Voronoi cells.
- **Why a graph.** Robots are nodes; an edge links two robots closer than the communication range
  R. Each robot only sees a local 64 m footprint, so it must learn about the rest of the world
  through messages.
- **LPAC** (Agarwal et al., IEEE T-RO 2025): CNN on four 32×32 local maps → 5 TAGConv layers
  (K = 3, 15 hops) → MLP giving a velocity; imitation of the clairvoyant controller. One clean
  architecture diagram, colour-coded: CNN path and GNN path (reuse the site's SVG).
- **The detail that matters later:** two of the four CNN channels are a *neighbour map*, an image
  of where the neighbours are, drawn at the scale of R.
- Short PyG code block: how the radius graph and TAGConv stack are built.

## 3. The anomaly (about 400 words)

- Figure 1: cost against range, published model, 11 released and 40 new environments.
- Table: the milestone Table 1 rows that matter (published, fixed map at inference, centralized
  CVT, V4 mixed).
- Ruling out the boring explanations: deterministic evaluation, the authors' own released
  checkpoint and environments, the same result on 40 environments we generated ourselves.

## 4. Diagnosis: is it the graph or the picture of the graph? (about 700 words)

- The idea: the neighbourhood enters the model twice, as the GNN's `edge_index` and as the
  neighbour map. Change one, keep the other: a 2×3 factorial at inference.
- Figure 2: factorial bar chart. Pruning the graph to the 4 nearest robots changes little; changing
  only the map restores the performance.
- Figure 3: what the neighbour map looks like at 128 m and at 1024 m (site map viewer as a static
  image). At 1024 m the map holds every robot of the team, drawn at 64 m per cell instead of
  8 m: the CNN receives an image unlike any it saw in training (exact wording to take from
  journal sections 4 and 9 when drafting).
- Dose-response: the cost tracks the scale of the map, not the number of edges.
- Takeaway box: *distribution shift in a node feature, not over-smoothing or over-squashing.*

## 5. Fixes (about 700 words)

- **Fix 1, free:** draw the map at a fixed 256 m scale and drop robots beyond it. The published
  model goes from 6.15 to 1.77 at 1024 m with no retraining. Clamping far robots onto the 256 m
  circle instead is worse (2.65): the TA's question Q1, answered with numbers.
- **Fix 2, retraining:** models that do not need the map. V4 replaces the map by bounded edge
  features on the graph, the unit direction to the neighbour and tanh(d / 256 m), aggregated by a
  max. Trained on mixed ranges, it is the only model whose cost falls as the range grows
  (1.87 → 1.81). Figure 4: the variants on one plot.
- Honest comparison: V4 mixed is +0.02 against LPAC retrained at 256 m, its best range. It wins by
  not needing to know the range in advance, not by being better at a given range.
- Short PyG code block: the V4 edge-feature layer (MessagePassing with max aggregation).

## 6. Two extensions (about 500 words)

- **Teams with different ranges.** When half the robots reach 1024 m and half 128 m, what is a
  link? Four rules compared; Figure 5. Transmission semantics helps V4 (−0.17 to −0.23 against
  reception, 11 of 11 environments) and hurts the published model.
- **Why GATv2 failed, and what repairs it.** A GATv2 variant stalls at a training loss of 0.676.
  Bounded edge features alone change nothing (V2B); residual blocks with LayerNorm repair it
  (0.432). Figure 6: training curves.
- Final-run results (replicates, V2M, transmission-trained teams) slot in here once evaluated.

## 7. Lessons for GNN practitioners (about 250 words)

1. Intervene on inputs one at a time; a factorial at inference is cheap and decisive.
2. Scale-dependent node features break transfer across graph densities; prefer bounded,
   scale-free features.
3. Pair comparisons by environment and report seeds; several of our first conclusions moved by
   more than the effect size until we did.

## 8. Try it yourself, and links (about 100 words)

- Colab badge and what it reproduces in how long.
- Project site (full details, interactive figures): https://loicpoisson.github.io/lpac-range-transfer/
- References: LPAC paper, CoverageControl library, PyG, TAGConv, GATv2.
- Team and course line.

---

## What stays out of the post

Kept on the site and in the journal only: effective-rank measurements, the constant-degree sweep,
V0R and V5/V6 details beyond one sentence, the checkpoint-selection study, engineering notes and
lost runs. The post links to the site for them.

## Figures to produce (from existing data, same palette as the site)

| # | Figure | Source |
|---|---|---|
| 0 | GIF: published at 1024 m, C0 against C3 | site/data/replays.json |
| 1 | Cost against range, published | costs.json, headline |
| 2 | Factorial at 1024 m | factorial.json |
| 3 | Neighbour map at 128 m and 1024 m | maps.json |
| 4 | Variants against range | costs.json |
| 5 | Mixed-range teams, four link rules | hetero.json |
| 6 | GATv2 training curves | curves.json |

## Colab (to match the post)

The existing `lpac_eval.ipynb` (October, French, published model only) installs from pip and runs
on a clean Python 3.12. Plan: rewrite it in English around the post: (1) install and download the
release, (2) reproduce the anomaly on 2 environments, (3) the factorial at 1024 m, (4) fix 1 at
inference, (5) load a V4 checkpoint and compare. Quick mode under 10 minutes on a free Colab CPU;
full mode documented. The V4 checkpoint (7.6 MB) is attached to a GitHub release of the public repo.
