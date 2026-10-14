# Project log

Decisions, measurements and corrections, in the order they happened. Numbers are costs
relative to the clairvoyant CVT (lower is better), averaged over the 11 environments released
with CoverageControl v1.3.1, 32 robots, 600 steps, unless stated otherwise. The detailed run
reports are the `RAPPORT*.md` files at the repository root (French).

## 1. Choosing the topic (early October)

**Course constraints.** Project type 1 (real-world application, Medium blog post + Colab):
applying an existing GNN is not enough, the rubric asks for novelty or thorough analysis.
Proposal due 20 October (NeurIPS'22 style, 2 pages, public data), milestone 10 November,
report 10 December. Course GCP credits: one code per student; GPU quota must be requested
and takes days.

**Topics considered.** Hyperspectral imaging with GNNs (link with the SLAC XRF work, dropped
because the SLAC data is not public), graph-based image compression (dropped: graph signal
processing rather than GNNs, crowded field), graph condensation (kept as a fallback), then
robotics, at the team's request:

| Option | Idea | Outcome |
|---|---|---|
| A | Learned physics simulators (GNS) for deformable objects | not chosen: PyG implementations already exist, no clear open question |
| B | Robustness of decentralized multi-robot path planning (MAGAT) to message loss, delay and robot failure | fallback: the gap is named by two papers, but the result is predictable, the code does not use PyG and the data sits on Google Drive links from 2020-2021 |
| C | Why LPAC's coverage performance degrades when the communication range grows | **chosen**: an open anomaly stated by the authors (Section VI-H), code in PyTorch Geometric, maintained library |

**Prior work check.** No paper found that evaluates LPAC, or a comparable swarm policy, at a
communication range larger than its training range (search of the Google Scholar list of
papers citing LPAC, plus targeted searches; three theses on that list are still unread:
Gosrich 2024, Owerko 2025, F. Yang 2026). Closest work: Muthusamy et al. 2024 (k-nearest
communication, policy improves with density at test time), MAST 2025 (larger attention
windows generalize worse, varied at training time), and the general GNN literature on
degree shift. Hugging Face, the authors' pages and their GitHub accounts host no other LPAC
checkpoint.

## 2. What the code actually is

- The GNN is five `torch_geometric.nn.TAGConv` layers with K = 3: the symmetrically
  normalized adjacency, applied up to K times per layer. It is not a sum aggregation, so
  hypothesis H2 of the paper ("summation is lossy") does not describe the implementation.
- `edge_weight` is built but not passed to the GNN: the adjacency is binary.
- Neighbor information enters the model twice: through the graph given to the GNN, and
  through a 32 × 32 neighbor map given to the CNN, whose resolution scales with the range.
- Only one checkpoint is public, `model_k3_1024_state_dict.pt`, byte-identical across
  releases v1.1.0 to v1.3.1. Its parameters indicate a training range of 128 m. The paper's
  Fig. 20 compares models trained at each range, so it cannot be reproduced with it.
- The repository's evaluators at HEAD are incompatible with the 1.3.1 wheel; everything
  uses tag v1.3.1. The wheel exists only for Linux, so the work runs in WSL and on FarmShare.

## 3. Range sweep of the published model (12 October)

| Range (m) | 128 | 256 | 512 | 768 | 1024 |
|---|---|---|---|---|---|
| LPAC-K3, published | 1.98 | **1.69** | 2.77 | 4.52 | 5.22 |
| Decentralized CVT | 2.59 | 2.57 | 2.59 | 2.59 | 2.59 |
| Centralized CVT | 2.52 | 2.52 | 2.52 | 2.52 | 2.52 |
| Neighbors per robot (LPAC) | 1.4 | 4.5 | 11.7 | 18.5 | 24.9 |

- The collapse appears in every environment (cost ratio 1024 m / 256 m between 2.1 and 4.6).
  At 768 and 1024 m the final cost exceeds the initial one.
- This is a transfer result: one model trained at 128 m, run at other ranges. It is not a
  reproduction of the paper's Fig. 20 (optimum at 512 m, mild decline, LPAC better than CVT
  everywhere, 100 environments).
- The decentralized CVT is flat only on average: it changes by −9 % to +5 % per environment
  between 128 and 512 m, and is exactly constant from 512 m on (it computes its cell in a
  256 m local window).
- Constant neighbor count (range up, team size down, 4 to 5 neighbors): the collapse
  disappears (1.7 to 2.0), but team size changes the task.
- Effective rank of the TAGConv outputs drops about 30 % with range in layers 2 to 4 but
  never collapses: over-smoothing is not supported.

## 4. Where the failure comes from (12 October)

Inference-time factorial at 1024 m, same weights (codes in the README):

| | C0 full map | C1 4 nearest | C2 128 m scale | C3 256 m scale |
|---|---|---|---|---|
| G0 full graph | 5.22 | 2.91 | 2.35 | **1.71** |
| G1 graph capped to 4 | 5.11 | 2.95 | 2.18 | **1.71** |

- Changing the graph alone does nothing measurable (−0.11, better in 5 of 11 environments).
  Changing the map recovers most or all of the loss, in 11 of 11 environments.
- With the map at the 256 m scale, the model at 1024 m matches its own 256 m performance
  (paired difference +0.02 ± 0.16) at 512, 768 and 1024 m.
- Empty map: 2.82 at 128 m and 2.42 at 256 m, so the map carries useful information. At
  1024 m an empty map (3.20) beats the full one (5.22), and with an empty map capping the
  graph helps (2.65): graph density has a real but secondary effect.
- Same neighbor count, different scale gives different results (about 4.5 neighbors: 1.69 at
  the 256 m scale, 2.91 at the 1024 m scale), so the map scale matters, not only the count.
- Noise floor: moving the initial positions by 1 m changes an episode's cost by about 7 %.
  Differences below about 0.1 on the 11-environment mean are not interpretable.
- Caveats: the 256 m scale was selected on the same 11 environments; the fix makes the model
  ignore neighbors beyond 256 m, so it gains nothing from a larger range.

## 5. Training is feasible (12-13 October)

- One "dataset" item of the generator is one sample: the state of the 32 robots at one time
  step. The paper's 100,000 data points are about 250 episodes. Disk: 0.37 MB per sample.
- Full paper scale (100,000 samples, 100 epochs) trains in 2 h 35 on a FarmShare L40S GPU,
  free. CPU-only is not realistic in under 6 h at that scale. The GPU queue is short at
  night, more than 3 h during the day.
- Best validation loss 0.449 at epoch 13, then overfitting; the best checkpoint is kept.
- Our retrained model matches the published one at 256 m (1.74 vs 1.69, paired +0.05) and
  beats the centralized CVT in 11 of 11 environments at 128 and 256 m.
- **It reproduces the collapse**: 2.15, 1.74, 2.49, 3.46, 4.36 from 128 to 1024 m. A second
  seed gives the same curve. It degrades less than the published checkpoint beyond 512 m,
  more than seed variability explains: the shape is robust, the magnitude depends on the
  training data or protocol.
- Reproducibility: bit-identical results across Intel CPUs, up to 5.6 % per environment on
  AMD nodes (those runs were discarded and recomputed); environment generation cannot be
  seeded, generated files must be kept.

## 6. Corrections made along the way

Statements made during planning that turned out wrong, and what replaced them:

- "The GNN uses sum aggregation, so GIN's expressivity argument applies": wrong, it is
  `TAGConv` with normalized adjacency.
- "Five layers are too shallow for over-smoothing": the effective depth is 5 × 3 = 15
  propagations; in any case the effective-rank measurement shows no collapse.
- "Use `knn_graph` as the degree-capped control": it would link robots out of radio range;
  the capped graph keeps the k nearest neighbors within range.
- "Measure over-smoothing with Dirichlet energy": replaced by effective rank, following
  Zhang et al., ICLR 2026, which shows Dirichlet-type measures fail for shallow networks.
- "No pretrained model is published": it is, attached to the GitHub releases.
- "Training data would need hundreds of terabytes": maps are stored at 32 × 32; 100,000
  samples take about 37 GB.
- "Over-smoothing (H3) is the favored explanation": the factorial points to the neighbor map
  (closer to the paper's H1), with graph density as a secondary effect.

## 7. A model trained at 256 m (14 October, 00:05)

Same recipe, 100,000 samples generated with `CommunicationRange = 256` (4 to 5 neighbors
per robot in the training maps and graphs). The GPU job hit its 3 h limit at epoch 35
(265 s per epoch, loader-bound); best validation loss **0.391 at epoch 7**, lower than at
128 m (0.449) since imitation is easier with more neighbors. Sweep, 11 environments:

| Range | 128 m | 256 m | 512 m | 768 m | 1024 m |
|---|---|---|---|---|---|
| trained at 256 m | 2.28 ± 0.50 | 1.76 ± 0.28 | **1.74 ± 0.17** | 1.90 ± 0.24 | **2.31 ± 0.24** |
| trained at 128 m (seed 0) | 2.15 ± 0.51 | 1.74 ± 0.25 | 2.49 ± 0.32 | 3.46 ± 0.71 | 4.36 ± 1.08 |
| paired (256 m − 128 m) | +0.13 ± 0.23 | +0.02 ± 0.23 | −0.75 (11/11) | −1.56 (11/11) | −2.06 (11/11) |

- As good as the 128 m model at its own range, slightly worse at 128 m (fewer neighbors
  than in training), and far more robust beyond: 2.31 at 1024 m instead of 4.4 to 5.2.
- The degradation is pushed back, not removed: 1.74 → 2.31 from 512 to 1024 m, and at
  1024 m it beats the centralized CVT in only 6 of 11 environments.
- Reading: the training range sets the neighborhood regime the map can read; widening it
  delays the collapse. Reference for phase 2: a variant without maps must reach at least
  2.3 at 1024 m, ideally stay flat. Files: `results_train/results_sweep256_best.csv`,
  `training/logs/loss_256m_gpu.json`, figure `fig_range_retrained.png`.

## 8. Seed spread and a decorrelated dataset (14 October, overnight)

**Third seed of the 128 m, 100k recipe** (best validation 0.450 at epoch 10). The three
seeds give 2.15 / 2.14 / 2.14 at 128 m, 1.74 / 1.73 / 1.76 at 256 m, 2.49 / 2.51 / 2.36 at
512 m, 3.46 / 3.54 / 3.74 at 768 m and 4.36 / 4.58 / 4.89 at 1024 m. Standard deviation
between seeds of the 11-environment mean: 0.01, 0.02, 0.08, 0.14, 0.26. Rule for phase 2:
a difference between two variants is meaningful above ~0.05 at 256 m and ~0.5 at 1024 m
(two standard deviations), otherwise repeat with seeds.

**Decorrelated dataset** (`EveryNumSteps = 20` instead of 5: one sample every 20 steps,
four times more episodes for the same 100,000 samples; generation time unchanged, about
1 080 s per 2 000 samples). Best validation 0.441 at epoch 7 (vs 0.449): slightly easier
to fit. The sweep, however, is a different animal:

| Range | 128 m | 256 m | 512 m | 768 m | 1024 m |
|---|---|---|---|---|---|
| every 20 steps | 2.24 ± 0.34 | 2.01 ± 0.24 | 2.87 ± 0.33 | **3.01 ± 0.41** | **3.02 ± 0.41** |
| every 5 steps (seed 0) | 2.15 ± 0.51 | 1.74 ± 0.25 | 2.49 ± 0.32 | 3.46 ± 0.71 | 4.36 ± 1.08 |
| paired (20 − 5) | +0.09 ± 0.25 | +0.27 ± 0.16 (0/11 better) | +0.38 ± 0.47 | −0.44 ± 0.76 | −1.35 ± 0.99 (11/11 better) |

Worse at its own range (2.01 vs 1.74 at 256 m, worse in 11 of 11 environments) and flat
beyond 768 m, around 3.0, where it beats the 5-step model but still not the CVTs (2 of 11).
The robots also cluster more (degree 29.6 vs 24.3 at 1024 m). Reading: sampling every
20 steps under-represents the early, far-from-equilibrium states that teach the policy
to spread out; the lower MSE does not mean a better controller. For comparisons with the
published model, keep the authors' recipe (every 5 steps). Files:
`results_train/results_sweep100k_seed2_best.csv`, `results_train/results_sweep_e20_best.csv`,
`training/logs/loss_100k_every20_gpu.json`, figure `fig_range_recipes.png`.

## 9. Running and next (as of 14 October)

Nothing running on FarmShare as of 10:30; all overnight chains retrieved (sections 7, 8).

Next, in order:

1. Fixed-scale map (C3) and empty map (C4) on our retrained models.
2. Validation on about 40 new environments (saved to disk) and a sweep of the map scale
   (128, 192, 256, 384, 512 m).
3. Information loss vs distribution shift in the map: same neighbors in a finer map.
4. The graph ML contribution: a variant without neighbor maps, where relative positions are
   carried by edge attributes of a PyG layer (for example `GATv2Conv` with `edge_dim=2`),
   trained at 128 m and tested across ranges.
5. Optionally, LPAC trained at 512 m to connect with Fig. 20.

Contact: an email with the results and three questions (other checkpoints, whether the
neighbor-map sensitivity is known, scope and compute) was sent to Charilaos Kanatsoulis,
co-instructor and co-author of Muthusamy et al., in the week of 12 October.
