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

## 9. The diagnosis holds on retrained models and on new environments (14 October)

Two CPU measures on FarmShare (about 1,500 episodes, 50 minutes wall with 40-task arrays).
Thresholds: a difference between two conditions is interpretable above two seed standard
deviations of the 11-environment mean (0.04 at 256 m, 0.16 at 512 m, 0.28 at 768 m,
0.52 at 1024 m, section 8), and above ~0.1 per environment (episode noise, M3).

**Measure 1: map levels on our four retrained models** (full graph, 512 / 768 / 1024 m,
11 release environments; C0 = the model's own full map from its sweep).

| Model | Range | C0 full map | C2 (128 m scale) | C3 (256 m scale) | C4 (empty) | best − C0, paired |
|---|---|---|---|---|---|---|
| 128 m, seed 0 | 512 / 768 / 1024 | 2.49 / 3.46 / 4.36 | 2.11 / 2.15 / 2.15 | **1.81 / 1.86 / 1.84** | 2.58 / 2.58 / 2.59 | −0.68 / −1.59 / −2.52 (10-11/11) |
| 128 m, seed 1 | 512 / 768 / 1024 | 2.51 / 3.54 / 4.58 | 2.10 / 2.10 / 2.08 | **1.76 / 1.82 / 1.83** | 2.77 / 2.74 / 2.76 | −0.76 / −1.72 / −2.75 (10-11/11) |
| 128 m, seed 2 | 512 / 768 / 1024 | 2.36 / 3.74 / 4.89 | 2.20 / 2.11 / 2.12 | **1.76 / 1.83 / 1.81** | 2.68 / 2.77 / 2.77 | −0.60 / −1.91 / −3.08 (10-11/11) |
| 256 m | 512 / 768 / 1024 | 1.74 / 1.90 / 2.31 | 2.42 / 2.52 / 2.46 | 1.87 / 1.89 / **1.89** | 2.82 / 2.84 / 2.83 | +0.13 / −0.01 / −0.42 (5-9/11) |

- The diagnosis "the map, not the graph" holds for every model we trained: with the map
  drawn at the 256 m scale, the three 128 m models sit at 1.76-1.86 at every range, which
  is their own 256 m level (1.73-1.76); the effect size (−0.6 to −3.1) is far above the
  seed threshold. The empty map is worse than any drawn map (2.6-2.8).
- The best scale is **256 m for all four models**, not the training range: for the 128 m
  models C3 beats C2 by 0.25-0.44 (7-10 of 11 environments); for the 256 m model C2 hurts
  (+0.55-0.62, 11/11) and C3 equals its own full map at 512 m (+0.13, within noise) and
  improves it at 1024 m (−0.42, 9/11, just under the 0.52 threshold: suggestive only).
  So 256 m looks like a property of the map encoding (4-5 robots drawn, one per cell),
  not of the checkpoint.

**Measure 2a: 40 new environments** (generated with the library's own generator and saved in
`environments/new40/`, never used before; CVT baselines recomputed on them).

| Condition | 128 m | 256 m | 512 m | 1024 m | same on the 11 release envs |
|---|---|---|---|---|---|
| published, full map | 1.96 ± 0.42 | 1.70 ± 0.23 | 2.87 ± 0.40 | **6.15 ± 1.45** | 1.98 / 1.69 / 2.77 / 5.22 |
| published, map at 256 m scale | 1.89 ± 0.36 ¹ | 1.70 ± 0.23 | 1.71 ± 0.22 | **1.77 ± 0.24** | 1.93 ¹ / — / 1.78 / 1.71 |
| 128 m seed 0, full map | 2.13 ± 0.37 | 1.82 ± 0.27 | 2.53 ± 0.32 | 4.76 ± 1.21 | 2.15 / 1.74 / 2.49 / 4.36 |
| 128 m seed 0, map at 256 m scale | 2.06 ± 0.35 ¹ | 1.82 ± 0.27 | 1.78 ± 0.21 | 1.80 ± 0.21 | 2.05 ¹ / — / 1.81 / 1.84 |
| 256 m model, full map | 2.42 ± 0.44 | 1.85 ± 0.34 | 1.81 ± 0.28 | 2.29 ± 0.25 | 2.28 / 1.76 / 1.74 / 2.31 |
| centralized / decentralized CVT | 2.33 / 2.54 | 2.33 / 2.45 | 2.33 / 2.44 | 2.33 / 2.44 | 2.52 / 2.6 |

- Every number of the release set is reproduced on the new set within its standard error,
  except the collapse of the published model at 1024 m, which is even deeper on new
  environments (6.15 vs 5.22, +0.93 with a standard error of 0.48).
- The fix generalizes: with the 256 m map, the published model is flat (1.70-1.77) from
  256 m up and beats the centralized CVT in 36 to 39 of 40 environments at those ranges;
  paired against its own 256 m optimum, the 1024 m cost is +0.06 ± 0.22 (standard error
  0.04), i.e. within noise. The retrained model with the same fix: 1.78-1.82, 37-39 of 40.
- ¹ At 128 m the "256 m scale" map can only contain the robots within radio range, so the
  condition differs from the as-is map only by the coarser drawing resolution. Corrected
  values (see the note below): published 1.89 ± 0.36 on the 40 environments, paired −0.07
  ± 0.30 against the as-is 1.96 (better in 21 of 40, within noise), 1.93 ± 0.25 on the 11
  release environments (as is: 1.98); retrained seed 0: 2.06 ± 0.35 (as is 2.13, −0.07 ±
  0.15, 29 of 40) and 2.05 ± 0.47 on the 11 (as is 2.15). The fixed-scale map therefore
  neither helps nor hurts below its scale: it is a fix for ranges above 256 m only.

**Correction of 14 October (afternoon).** The first version of this table reported 1.72
(published) and 1.85 (seed 0) at 128 m with the 256 m map. Those numbers were invalid:
the mask that selects the robots to draw used the map scale alone, so at a test range of
128 m the map contained robots up to 256 m away, which the robot cannot hear at that
range. `FactorialPolicy._map_mask` now draws the robots within min(test range, scale) at
the resolution of the scale; at ranges at or above the scale nothing changes (the 256 m
control is still bit-identical to the as-is run, M1), and the 128 m points were recomputed
(`results_fix_c3_128/`). The 1024 m scale sweep (measure 2b) is unaffected, since every
scale there is below the test range.

**Measure 2b: map scale at 1024 m, published model.** 128 / 192 / 256 / 384 / 512 m:
2.36 / 2.00 / **1.71** / 2.16 / 2.78 on the 11 release environments, 2.23 / 1.80 / **1.77** / 2.07
/ 2.96 on the 40 new ones. The optimum is at 256 m on both sets (192 and 256 are tied on the
new set, 1.80 vs 1.77), well below the 2.3-2.5 of the CVTs; 512 m is already worse than
the CVTs. The scale was therefore not chosen on the test environments by luck.

Figures: `fig_map_levels_retrained.png`, `fig_new_envs.png`, `fig_map_scale.png`. Data:
`results_m1/results_2x2.csv` (model levels and the 11-env scale sweep), `results_new40/`
(sweeps and factorial on the 40 environments), `environments/new40/` (the environments).

## 10. Running and next (as of 14 October)

Nothing running on FarmShare as of 10:30; all overnight chains retrieved (sections 7, 8).

Next, in order (1 and 2 done, section 9):

3. Information loss vs distribution shift in the map: same neighbors in a finer map.
4. The graph ML contribution: a variant without neighbor maps, where relative positions are
   carried by edge attributes of a PyG layer (for example `GATv2Conv` with `edge_dim=2`),
   trained at 128 m and tested across ranges.
5. Optionally, LPAC trained at 512 m to connect with Fig. 20.

Contact: an email with the results and three questions (other checkpoints, whether the
neighbor-map sensitivity is known, scope and compute) was sent to Charilaos Kanatsoulis,
co-instructor and co-author of Muthusamy et al., in the week of 12 October.
