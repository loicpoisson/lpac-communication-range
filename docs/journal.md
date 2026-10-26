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

## 10. Variants without neighbor maps (14 October, started)

The diagnosis says the neighbor map is what breaks when the range grows, because its
resolution follows the range. The graph ML question is whether the GNN can carry the
neighbor geometry itself, so that nothing in the input depends on the range. Three models,
all trained with the authors' recipe (100,000 samples at 128 m, 100 epochs, best validation
checkpoint), three seeds each:

| Variant | CNN input | GNN | Parameters | Receptive field |
|---|---|---|---|---|
| V0 (as published) | 4 channels: local map, two neighbor-map channels, obstacle map | 5 × `TAGConv`, K = 3, width 256 | 1,806,210 | 15 hops |
| V1 | 2 channels: local map, obstacle map (`UseCommMaps = false`) | unchanged | 1,805,634 | 15 hops |
| V2 | as V1 | 5 × `GATv2Conv`, `edge_dim = 2`, 4 heads × 64 concatenated (width 256), ReLU | 1,270,338 | 5 hops |

V2 reads on every edge (j → i) the relative position (p_j − p_i) / W, W the world size, computed
in the forward pass from `data.pos` and `data.edge_index`; `data.pos` is already (p + W/2) / W, so
the attribute is a constant of the world, not of the range. V1 removes the map without replacing
it: it isolates what the map contributes at 128 m (V0 − V1) from what the edge geometry adds
(V2 − V1). The library model is subclassed, not modified (`lpac_study/variants.py`); the
dataset is read with `use_comm_map = False`, which gives 2-channel inputs and keeps `pos` and
`edge_index`. V2 has fewer parameters because a GATv2 layer with 4 heads of 64 holds one
256 × 64 matrix per head and no K-hop filters; its receptive field is 5 hops instead of 15.

Comparisons planned, with the seed thresholds of section 8 (std of the 11-environment mean
across seeds: 0.01 / 0.02 / 0.08 / 0.14 / 0.26 at 128 / 256 / 512 / 768 / 1024 m):
V2 against V0 as is (collapse or not), against V0 with the 256 m map (1.81-1.84 at 1024 m) and
against the model trained at 256 m (2.31), and V2 against V1 (does the edge geometry help).
Ranges 128 to 1024 m on the 11 release and the 40 new environments.

Pipeline trial (14 October, 13:00): 2,000 samples (one generation part, split 1,400 / 400 / 200),
3 epochs, then one evaluation episode at 256 m on environment 0. The GPU partition was
backed up (estimated start the next night), so the trial ran on an Intel CPU node with
16 cores: 34 s per epoch for both variants, validation loss 0.700 → 0.694 (V1) and
0.700 → 0.696 (V2), evaluation cost ratio 2.73 (V1) and 2.73 (V2) after 3 epochs on
1,400 samples, i.e. the whole chain (dataset with 2 channels, variant model, checkpoint,
evaluation through `lpac_study.sweep --variant`) works end to end.

Full runs submitted at 13:05: V1 and V2 × seeds 0, 1, 2 on the GPU partition (jobs
1846008-1846025, 100 epochs, best-validation checkpoint, each chained with the evaluations
on the 11 release and the 40 new environments at 128 / 256 / 512 / 768 / 1024 m), plus a
CPU hedge for seed 0 of each variant (12 epochs on 32 Intel cores, jobs 1846026-1846031)
in case the GPU queue does not clear overnight. The GPU queue cleared at 17:15; all eight
trainings and their evaluations finished by 23:00.

**Training.** 54-65 s per epoch on an L40S, about 2 h per run. Best validation loss (the
checkpoint kept): V1 0.536 / 0.528 / 0.543 at epochs 18 / 17 / 26, then overfitting (train
0.21, validation 0.64 at epoch 100); V2 0.676 / 0.678 / 0.677 at epochs 23 / 22 / 22, with
no overfitting (train 0.62 at epoch 100): V2 underfits. The V0 reference reaches 0.449 at
epoch 13. The 12-epoch CPU runs land on the same curves (V1 0.546, V2 0.681) and give the
same costs within seed noise; they are kept as a fourth point in `results_variants/`.

**Results** (cost ratio, environment mean, mean ± std across the 3 seeds; V0 = our three
retrained seeds; V0 with the 256 m map from section 9; centralized CVT 2.52 / 2.33).

| 11 release environments | 128 m | 256 m | 512 m | 768 m | 1024 m |
|---|---|---|---|---|---|
| V0 as published | 2.15 ± 0.01 | **1.74 ± 0.02** | 2.46 ± 0.08 | 3.58 ± 0.14 | 4.61 ± 0.26 |
| V1 no neighbor map | 2.49 ± 0.08 | 2.48 ± 0.04 | 2.80 ± 0.13 | 2.98 ± 0.18 | 3.14 ± 0.20 |
| V2 edge geometry (GATv2) | 2.51 ± 0.00 | 2.96 ± 0.02 | 3.25 ± 0.06 | 3.28 ± 0.06 | 3.31 ± 0.07 |
| V0 with the map at 256 m scale | = V0 | 1.74 ± 0.02 | 1.78 ± 0.03 | 1.84 ± 0.02 | 1.83 ± 0.02 |
| V0 trained at 256 m (1 seed) | 2.28 | 1.76 | 1.74 | 1.90 | 2.31 |

| 40 new environments | 128 m | 256 m | 512 m | 768 m | 1024 m |
|---|---|---|---|---|---|
| V0 as published (seed 0) | 2.13 | **1.82** | 2.53 | — | 4.76 |
| V1 no neighbor map | 2.39 ± 0.06 | 2.54 ± 0.03 | 2.84 ± 0.13 | 3.02 ± 0.19 | 3.12 ± 0.23 |
| V2 edge geometry (GATv2) | 2.57 ± 0.02 | 2.97 ± 0.07 | 3.26 ± 0.07 | 3.27 ± 0.07 | 3.29 ± 0.06 |
| V0 with the map at 256 m scale (seed 0) | 2.06 | 1.82 | 1.78 | — | 1.80 |
| V0 trained at 256 m | 2.42 | 1.85 | 1.81 | — | 2.29 |

Paired, seed-matched differences (mean ± std across environments, environments where the
first is better), 11 release environments, same picture on the 40 new ones in brackets:

- V2 − V0 as is: +1.21 to +1.24 ± 0.5 at 256 m, 0 of 11 better, every seed [+1.07, 1/40];
  −1.07 to −1.64 at 1024 m, 9-11 of 11 [−1.54, 36/40]. V2 does not collapse, but it starts
  far above V0 and never reaches it below 768 m.
- V1 − V0 as is: +0.72 to +0.79 ± 0.3 at 256 m, 0 of 11 [+0.75, 0/40]: removing the map
  costs 0.75 at the training regime, the largest effect measured on this model apart from
  the collapse itself. −1.05 to −1.69 at 1024 m [−1.59, 38/40].
- V2 − V1: +0.44 to +0.51 ± 0.6 at 256 m, 1-3 of 11 better [+0.33, 6/40]; +0.34 to +0.59
  at 512 m; −0.02 to +0.46 at 1024 m (4 of 11) [+0.06, 17/40]. The edge geometry makes the
  model worse than no geometry at all between 256 and 768 m and equal at 1024 m.
- V2 − V0 with the 256 m map: +1.39 to +1.56 ± 0.5 at 512-1024 m, 0 of 11 [+1.42 to
  +1.48, 0/40]. V2 − V0 trained at 256 m: +1.21 at 256 m, +0.99 at 1024 m, 0-1 of 11.
- Against the centralized CVT: V1 ties at 128 and 256 m (−0.02, +0.00) and loses beyond;
  V2 ties at 128 m only (−0.02) and loses by 0.45-0.77 from 256 m [0.57-0.93 on 40].
- The seed spread of V1 and V2 (std 0.00-0.23) is at or below the V0 thresholds, so all the
  differences above are far outside seed noise.

**Reading.** The rasterized neighbor map is not a redundant input: without it the model
loses 0.75 at 256 m and never beats the centralized CVT beyond 128 m. Carrying the same
geometry on the edges of a 5-layer GATv2 (V2) does not replace it with the authors' recipe:
V2 underfits (validation 0.68 against 0.53 for V1 and 0.45 for V0) and is worse than V1 at
every range up to 768 m. Both map-less variants are flatter across ranges (3.1-3.3 at
1024 m instead of 4.6) because they have less to lose, not because they are better: the
range-dependent input is gone, but so is most of the useful information. The two fixes
that work remain the inference-time map at a fixed 256 m scale (1.8 at every range from
256 m, section 9) and training at 256 m (2.3 at 1024 m, section 7).

Caveats. (1) V2 uses the release hyperparameters (learning rate 1e-4, weight decay 1e-3,
batch 100, 5 layers, no normalization or residual connection); a GATv2 stack may need its
own, and the underfitting suggests it. One architecture and one recipe were tried, so the
result is "this edge-geometry variant does not work out of the box", not "edge geometry
cannot work". (2) V2's edge attributes are (p_j − p_i) / W: at training they span ±0.125
(128 m over a 1024 m world), at 1024 m they span ±1, so the distribution shift the map
suffers from is still present in V2, on the edges; its flat curve means the attention does
not break, not that the shift is absent. (3) Receptive field 5 hops against 15: at 128 m
(mean degree 1.9, isolated fraction 0.31) the graph is fragmented and depth matters little;
at 1024 m (degree 30) one hop already reaches everyone.

Figures: `fig_variants.png` (11 release environments), `fig_variants_new40.png` (40 new
ones). Data: `results_variants/<variant>_s<seed>_{11,40}.csv` (merged sweeps), `train_*/`
(loss histories, timings, run info), `configs/`.

## 11. Night of 14-15 October: mixed-range training, capped graph, V3 (submitted 00:40)

Three experiments queued on FarmShare, in this order of priority.

**1. V0 trained at mixed ranges.** 100,000 samples with the authors' recipe (one sample
every 5 steps), in four equal parts of 25,000 at CommunicationRange = 128, 256, 512 and
1024 m (10 generation tasks of 2,500 samples per range, jobs 1847928-1847931), merged with
`merge_datasets.py` (job 1847991), then V0 trained with the release hyperparameters and the
best-validation checkpoint: seed 0 first (job 1847992), seeds 1 and 2 if the GPU queue
allows (1848015, 1848018), each chained with the evaluations at the five ranges on the 11
release and the 40 new environments. Question: does training on varied ranges remove the
collapse without losing the 256 m quality? Comparison with the seed thresholds against the
128 m model (as is and with the 256 m map) and the 256 m model.

**2. Graph capped at 256 m at inference for V1 and V2** (new GRAPH level `R<m>` in
`lpac_study/factorial.py`: edges only between robots within min(m, test range)). The local
check before launching showed that for the map-less variants this measure is an identity:
their only range-dependent input is the graph, so V2 at 1024 m with edges capped at 256 m
is bit for bit V2 at 256 m (cost 2.7060 = 2.7060 on environment 0, 30 steps; same for V1
and for the 128 m cap). The 2,754 planned episodes would therefore only recompute the
256 m and 128 m columns of section 10. Launched instead: the bit-exact confirmation over
full episodes on the 11 release environments at 1024 m, V1 and V2 seed 0, caps 256 and
128 m (jobs 1847989-1847990, 22 episodes, 2 min each). Result: V1 2.5289 against 2.5292
as is at 256 m, V2 2.9715 against 2.9715; per environment the two runs are equal to the
last bit in 5 to 8 of 11 cases and differ by at most 0.006 in the others (the factorial
and the sweep build the edge list in a different order, which changes the floating-point
summation order inside TAGConv and lets the 600-step trajectories drift slightly). Same
with the 128 m cap (2.505 against 2.506). The 3-seed, 40-environment numbers are therefore
those of section 10 at 256 m (V1 2.48 / 2.54, V2 2.96 / 2.97), data in
`results_night2/cap_V*_s0_11.csv`. Answer to the question asked: capping the graph
puts V2 back exactly on its 256 m value, so the whole degradation of V2 from 256 to
1024 m (2.96 to 3.31) comes from the extra edges, whose attributes lie outside the training
range; for V1 (2.48 to 3.14) it comes from the graph density alone. Capping does not make
either variant good: 2.96 and 2.48 remain above the centralized CVT (2.52) or at it.

**Result of experiment 1 (16 October, 01:00; provisional: seed 0, CPU, 7 epochs).** The GPU
queue did not clear (section 12), so the result comes from the CPU hedge: 12 epochs planned
on 32 Intel cores, stopped by the 11 h limit after 7 epochs (80 min per epoch on a shared
node), best validation loss 0.532 at epoch 5, then rising (0.56 at epochs 6-7) while the
training loss keeps falling, so the checkpoint kept is probably near what the recipe gives.
The validation loss is on the mixed validation set and is not comparable with the 128 m
runs. Cost ratio, environment mean (std across environments in brackets):

| | 128 m | 256 m | 512 m | 768 m | 1024 m |
|---|---|---|---|---|---|
| mixed ranges, seed 0 (11 release envs) | 2.55 (0.62) | 2.11 (0.33) | **1.87** (0.20) | 1.94 (0.20) | 1.91 (0.21) |
| 128 m model, seed 0, as is | 2.15 | 1.74 | 2.49 | 3.46 | 4.36 |
| 128 m model, seed 0, map at 256 m scale | 2.05 | = as is | 1.81 | 1.86 | 1.84 |
| 256 m model | 2.28 | 1.76 | 1.74 | 1.90 | 2.31 |
| mixed ranges, seed 0 (40 new envs) | 2.56 (0.47) | 2.22 (0.37) | **2.01** (0.30) | 2.04 (0.32) | 2.02 (0.30) |
| 128 m model, seed 0, as is (40) | 2.13 | 1.82 | 2.53 | — | 4.76 |
| 128 m model, map at 256 m scale (40) | 2.06 | 1.82 | 1.78 | — | 1.80 |
| 256 m model (40) | 2.42 | 1.85 | 1.81 | — | 2.29 |

Paired against the same environments (mean ± std, environments where the mixed model is
better), 11 release environments [40 new ones]:

- mixed − 128 m model as is: +0.40 ± 0.37 (1/11) at 128 m, +0.38 ± 0.19 (0/11) at 256 m,
  −0.62 (10/11) at 512 m, −2.45 ± 1.06 (11/11) at 1024 m [+0.43, +0.39, −0.51, −2.73 (39/40)].
- mixed − 128 m model with the 256 m map: +0.06 / +0.08 / +0.07 ± 0.3 at 512 / 768 /
  1024 m (3-6 of 11, within noise) [+0.24 / +0.22 ± 0.20, 5 of 40: slightly worse];
  +0.51 at 128 m (0/11).
- mixed − 256 m model: +0.35 ± 0.23 (0/11) at 256 m, −0.39 ± 0.17 (11/11) at 1024 m
  [+0.37 (1/40), −0.27 (29/40)].
- mixed − centralized CVT (2.52 / 2.33): −0.41 to −0.66 from 256 m (8-10 of 11)
  [−0.11 to −0.31, 26-34 of 40]; equal at 128 m.

Answer to the question. Training on mixed ranges removes the collapse: the curve is flat
from 512 m (1.87-1.94 on the release set, 2.01-2.04 on the new one, spread 0.5-0.7 over
the whole range instead of 2.6-2.9 for the 128 m model), and it beats the centralized CVT
at every range from 256 m. It does not keep the 256 m quality: 2.11 against 1.74 for the
128 m model and 1.76 for the 256 m model, 0 of 11 and 2 of 40 environments better, far
outside the seed noise (0.02 at 256 m). Its best range is 512 m (1.87), and it is worse at
128 m than every single-range model (2.55). Against the inference-time fix (128 m model
with the 256 m map, 1.74 / 1.78 / 1.84 / 1.83 at 256-1024 m) it is equal beyond 512 m on the
release set and 0.2 worse on the new set, and 0.4 worse at 256 m. So, with the authors'
recipe, mixing ranges in the training data trades the 256 m optimum for robustness; the
fixed-scale map gets the robustness without the trade.

Caveats. One seed, 7 epochs on CPU with the best checkpoint at epoch 5 (the 128 m runs
reach their best at epochs 10-13, and the 12-epoch CPU 128 m model gave 2.17 / 1.76, within
0.02 of the GPU one, so 12 epochs are enough for that recipe; for the mixed recipe the
validation loss had already turned up at epoch 6, which argues against a large gain from
more epochs, but the 100-epoch GPU runs with three seeds (jobs 1848258, 1848261, 1848264,
estimated start 17 October) will settle it). The mixed validation set spans the four ranges,
so a per-range breakdown of the loss would be needed to see where the model gives up
quality; the costs above are that breakdown in the end.

**Three GPU seeds (17 October).** The GPU jobs ran on 16-17 October and hit their 3 h 30
limit after 32 epochs: an epoch of the mixed dataset takes 327 s on an L40S against 54-65 s
for the 128 m dataset, because the samples at 512 and 1024 m have 15-30 neighbours per
robot instead of 2. Best validation at epochs 7 / 4 / 4 (0.531 / 0.525 / 0.520), then
rising with the training loss still falling (0.19-0.21 at epoch 32): the recipe overfits
the mixed set early, and the best checkpoints are well inside the 32 epochs. Cost ratio,
mean ± std across the three seeds of the environment mean:

| | 128 m | 256 m | 512 m | 768 m | 1024 m |
|---|---|---|---|---|---|
| mixed ranges, 3 seeds, 11 release envs | 2.40 ± 0.03 | 2.08 ± 0.05 | **1.91 ± 0.12** | 1.97 ± 0.05 | 1.93 ± 0.09 |
| mixed ranges, 3 seeds, 40 new envs | 2.43 ± 0.04 | 2.15 ± 0.03 | **2.01 ± 0.03** | 2.01 ± 0.01 | 2.00 ± 0.07 |

Seed-matched paired differences on the 11 release environments: mixed − V0 as is
+0.23 to +0.30 at 128 m and +0.28 to +0.41 at 256 m (0-2 of 11 better, every seed),
−0.48 to −0.59 at 512 m (10-11 of 11), −2.5 to −3.0 at 1024 m (11 of 11); mixed − V0 with
the 256 m map +0.04 to +0.27 at 512-1024 m (1-5 of 11; seed 1 is the worst, +0.20 ± 0.14)
[40 envs, seed 0: +0.19 to +0.27, 6-7 of 40]; mixed seed 0 − 256 m model +0.30 at 256 m,
−0.42 at 1024 m (11 of 11). The CPU result stands in every respect: no collapse, 0.3 lost
at 256 m, equal to the fixed-scale map beyond 512 m on the release set and 0.2 behind it on
the new set, worse than every single-range model at 128 m. Data
`results_night2/mixed_s{0,1,2}_{11,40}.csv`, loss histories `results_night2/train_models_s*/`.

Figures: `fig_mixed_range.png` (11 release environments) and `fig_mixed_range_new40.png`
(40 new ones, three GPU seeds). Data of the CPU run: `results_night2/mixed_s0_cpu_{11,40}.csv`, loss history and timing in
`results_night2/train_models_s0_cpu/`, dataset parameters `results_night2/*.toml`
(generation: 40 tasks of 2,500 samples, 22-24 min each; merge 8 min, 65 GB; training 80
min per epoch on a shared 32-core Intel node, 7 epochs in 11 h).

**3. V3 = V1 + one `PointNetConv` layer in front of the unchanged TAGConv stack**
(`LPACPointNet` in `lpac_study/variants.py`, every choice commented in the class
docstring). The layer reads ``[x_j, (p_j - p_i) / 256 m]`` on every edge, max-aggregates,
and its output is added to the TAGConv input (residual); 1,889,636 parameters (V1 +
84,002), receptive field 16 hops. Seed 0 only, 100k samples at 128 m, queued after the
mixed seed-0 training (job 1848021) with its evaluations (1848022-1848023). Control before
reading the evaluation: the training loss must reach V1's level (about 0.2 at epoch 100);
if not, seeds 1 and 2 stay unlaunched and it is reported as such.

**Result of experiment 3 (16 October, 04:30; seed 0, CPU, 11 epochs).** Same CPU hedge as
experiment 1 (job 1849304, 32 Intel cores, 50 min per epoch, 11 epochs in 11 h), best
validation 0.480 at epoch 11. Training-loss control against V1 under the same recipe
(12 CPU epochs): V3 0.470 at epoch 11 against V1 0.534 at epoch 11 (0.525 at 12), and
validation 0.480 against 0.546; the control passes, so seeds 1 and 2 are queued on the
GPU partition (jobs 1851374 and 1851377, behind the mixed-range runs). V1 at 100 GPU
epochs ends at a training loss of 0.215, which 11 CPU epochs cannot be compared with.

| | 128 m | 256 m | 512 m | 768 m | 1024 m |
|---|---|---|---|---|---|
| V3 seed 0, 11 release envs (std across envs) | 2.21 (0.43) | 2.21 (0.25) | 3.39 (0.57) | 4.86 (2.15) | 5.83 (2.71) |
| V1 seed 0, same CPU recipe | 2.43 | 2.56 | 2.76 | 2.90 | 3.00 |
| V1 seed 0, 100 GPU epochs | 2.51 | 2.53 | 2.85 | 3.13 | 3.31 |
| V0 seed 0, as is | 2.15 | 1.74 | 2.49 | 3.46 | 4.36 |
| V0 seed 0, map at 256 m scale | 2.05 | = as is | 1.81 | 1.86 | 1.84 |
| V3 seed 0, 40 new envs | 2.17 (0.35) | 2.18 (0.23) | 3.39 (0.66) | 4.68 (1.41) | 5.67 (1.82) |
| V1 seed 0, same CPU recipe (40) | 2.52 | 2.59 | 2.72 | 2.84 | 2.87 |
| V0 seed 0, as is (40) | 2.13 | 1.82 | 2.53 | — | 4.76 |

Paired (mean ± std across environments, environments where V3 is better), 11 release
environments [40 new ones]:

- V3 − V1 (same CPU recipe): −0.22 ± 0.18 (9/11) at 128 m, −0.35 ± 0.34 (10/11) at 256 m
  [−0.35 (38/40), −0.42 (38/40)]; +0.63 (1/11) at 512 m, +1.97 at 768 m, +2.84 ± 2.23
  (0/11) at 1024 m [+0.68, +1.84, +2.80 (3/40)]. Same signs against the 100-epoch V1.
- V3 − V0 as is: +0.06 (3/11) at 128 m, +0.47 ± 0.15 (0/11) at 256 m, +1.47 ± 2.44 (2/11)
  at 1024 m [+0.04, +0.35, +0.91 (8/40)].
- V3 − V0 with the 256 m map: +1.58 / +3.00 / +3.99 at 512 / 768 / 1024 m, 0 of 11 [0 of 40].

Reading. Unlike V2, the geometry on the edges helps where it was trained: at 128 and
256 m V3 is better than V1 by 0.2-0.4 in 9-11 of 11 and 38 of 40 environments, and it is
the first map-less variant that comes close to V0 at 128 m (2.21 against 2.15). It is
still 0.47 above V0 at 256 m, so the PointNet layer recovers roughly half of what the map
gives at that range. And it collapses with the range exactly as V0 does, even more
steeply (5.8 against 4.4 at 1024 m, with a huge spread across environments: 2.7): the
relative positions the layer reads span ±0.5 units of 256 m at training and ±4 at 1024 m,
and the max over 25-30 neighbours of a network that never saw such inputs is unbounded.
This is the prediction made in section 10 for V2's edge attributes, now observed on a
layer that actually uses them: the collapse is not a property of the rasterized map but
of any input whose support grows with the range. The flat variants (V1, V2) are flat
because they ignore the geometry; the variants that read it (V0 through the map, V3
through the edges) collapse. As for V1 and V2, capping V3's graph at 256 m at inference
would put it back on its 256 m value (2.21) at every range, by the identity of experiment
2, which is still worse than V0 with the 256 m map (1.8).

The mean degree under V3 is 25 at 1024 m against 29.5 under V1: the collapsed policy
pushes robots apart, which is also what V0's collapse does (section 4).

Caveats written at the time: one seed, 11 CPU epochs (the validation loss was still
falling), so the 128-256 m numbers may improve with the 100-epoch GPU runs; "the collapse
beyond 512 m will not disappear with more epochs, since it comes from inputs outside the
training support". The second half of that sentence turned out to be wrong, see below.

**Three GPU seeds, 100 epochs (17 October): V3 does not collapse once trained.** Jobs
1848021, 1851374, 1851377 (2 h 09 to 2 h 25 each); best validation 0.443 / 0.439 / 0.441 at
epochs 23 / 23 / 22, the level of V0 (0.449) and far below V1 (0.53); training loss 0.38
at the best epoch, 0.20-0.21 at epoch 100. Cost ratio, mean ± std across seeds:

| | 128 m | 256 m | 512 m | 768 m | 1024 m |
|---|---|---|---|---|---|
| V3, 3 seeds, 11 release envs | 2.24 ± 0.06 | **2.01 ± 0.06** | 2.31 ± 0.16 | 2.49 ± 0.22 | 2.58 ± 0.29 |
| V3, 3 seeds, 40 new envs | 2.13 ± 0.01 | **2.03 ± 0.08** | 2.25 ± 0.25 | 2.39 ± 0.32 | 2.44 ± 0.35 |
| V3 seed 0, 11 CPU epochs (above) | 2.21 | 2.21 | 3.39 | 4.86 | 5.83 |
| V0, 3 seeds (section 10) | 2.15 ± 0.01 | 1.74 ± 0.02 | 2.46 ± 0.08 | 3.58 ± 0.14 | 4.61 ± 0.26 |
| V1, 3 seeds (section 10) | 2.49 ± 0.08 | 2.48 ± 0.04 | 2.80 ± 0.13 | 2.98 ± 0.18 | 3.14 ± 0.20 |

Seed-matched paired differences, 11 release environments [40 new ones, seed 0]:

- V3 − V1: −0.10 to −0.34 at 128 m, −0.46 to −0.48 at 256 m, −0.36 to −0.68 at 512 m,
  −0.39 to −0.77 at 1024 m, 8-10 of 11 at every range [−0.26 to −0.45, 30-36 of 40].
  V3 beats V1 everywhere.
- V3 − V0 as is: +0.04 to +0.15 at 128 m (2-3 of 11), +0.23 to +0.34 at 256 m (0-2 of 11),
  about 0 at 512 m, −0.7 to −1.4 at 768 m and −1.4 to −2.5 at 1024 m (9-11 of 11)
  [−0.01, +0.29, +0.01, —, −1.92 (36 of 40)].
- V3 − V0 with the 256 m map: +0.45 to +0.69 at 512 m, +0.52 to +0.88 at 768 m, +0.57 to
  +1.08 at 1024 m, 0-1 of 11 [+0.76, +1.04, 0-1 of 40].
- V3 seed 0 − centralized CVT: −0.34 and −0.45 at 128-256 m (8-10 of 11), −0.03 at 512 m,
  +0.22 and +0.40 at 768-1024 m (2-4 of 11) [−0.21, −0.21, +0.21, —, +0.51]. Seed 0 is the
  worst seed at long range (2.92 at 1024 m, one environment at 4.8); seeds 1 and 2 give 2.40
  and 2.43 and tie with the CVT there.

Corrected reading. The collapse of the CPU model (5.8 at 1024 m) was an artefact of
under-training: at 11 epochs the PointNet layer was unbounded outside its training support,
at 100 epochs it is not, and the fully trained V3 degrades only mildly with the range
(2.0 → 2.6 against 1.74 → 4.6 for V0 and 2.48 → 3.14 for V1). So reading the neighbour
geometry on the edges at a fixed scale, in front of an unchanged TAGConv stack, gives the
best map-less model of this study: it matches V0 at 128 m, stays 0.25-0.35 behind it at
256 m (0-2 of 11 environments better: the rasterized map still carries something the layer
does not), and avoids most of the collapse. It does not reach the inference-time fix (V0
with the 256 m map, 1.74-1.84 at every range), which remains the best option by 0.5-1.1
beyond 512 m. The statement of 16 October that "any input whose support grows with the
range breaks" is therefore too strong: it holds for the rasterized map and for an
under-trained edge layer, not for a trained one. The earlier V2 result (GATv2, underfits)
is unchanged.

Figures `fig_variants.png` and `fig_variants_new40.png` (V3 as a three-seed curve); data
`results_variants/V3_s{0,1,2}_{11,40}.csv`, loss histories `results_variants/train_models100k_V3_s*/`;
the CPU run stays in `results_variants/V3_s0_cpu_{11,40}.csv` and `train_V3_s0_cpu/`.

Generation and training times are read from the Slurm accounting and the job logs.

Incident at 00:45: 25 of the 40 generation tasks died at the saving step (exit code 1, no
traceback) because the home directory hit its quota (about 50 GB of compressed blocks; the
raw parts of the three earlier datasets were still on disk next to their merged copies).
Freed 24 GB (raw parts already merged, per-epoch checkpoints except best and last, two
superseded merged datasets), regenerated the 22 incomplete parts (jobs 1848253-1848256,
24 GB memory limit) and resubmitted the merge (1848257) and the training chain
(1848258-1848282). One 2,500-sample part takes 22-24 min on one Intel core pair
(0.54 s per sample), 5.3-5.9 GB of RAM.

## 12. Heterogeneous communication ranges at inference (17 October)

Loïc's question: what if the robots of one team do not all hear as far? Inference only,
published model, 11 release environments, 32 robots, `lpac_study/heterogeneous.py` and
`farmshare_hetero.sbatch` (11 CPU tasks, 85 s per episode, jobs 1854506 and 1854518).

Setting. A fraction f ∈ {0, 1/4, 1/2, 3/4, 1} of the robots gets the 1024 m range, the
others 128 m; the long-range subset is drawn once per environment with a fixed seed, two
draws per environment for the intermediate fractions (176 episodes). Reception semantics:
robot i knows the robots within its own range r_i, so the graph has an edge j → i when
|p_j − p_i| < r_i (directed when ranges differ) and the map of robot i is drawn with those
robots at the scale r_i. Second map mode, the fix of section 9: the map of robot i holds the
robots within min(r_i, 256 m) at the 256 m scale, graph unchanged. The simulator runs with
the long range as its parameter; only the model inputs are rebuilt. Checks: f = 0 reproduces
the 128 m sweep (1.985, max difference 1e-5), f = 1 with the fixed map reproduces the C3
run at 1024 m exactly (1.707); f = 1 with the as-is map gives 5.30 against 5.22 in the sweep
(one environment differs by 0.49): the edge list is built in another order than the
library's, and in the collapsed regime the 600-step trajectories amplify the floating-point
difference, which is itself a measure of how unstable that regime is.

| Fraction of robots at 1024 m | 0 | 1/4 | 1/2 | 3/4 | 1 |
|---|---|---|---|---|---|
| map at each robot's own range (as published) | 1.99 ± 0.46 | 2.36 ± 0.30 | 2.77 ± 0.58 | 3.77 ± 0.67 | 5.30 ± 1.51 |
| map at a fixed 256 m scale | 1.93 ± 0.25 | 2.01 ± 0.28 | 1.96 ± 0.39 | 1.84 ± 0.17 | 1.71 ± 0.19 |
| probe: relative action change of the 128 m robots, as-is map | — | 0.18 | 0.29 | 0.35 | — |
| probe: same, fixed 256 m map | — | 0.12 | 0.23 | 0.31 | — |
| probe: relative action change of the 1024 m robots, as-is / fixed | — | 1.16 / 0.87 | 1.21 / 0.89 | 1.28 / 0.88 | — |

(Cost ratio, mean ± std across environments of the per-environment mean over the two draws;
centralized CVT 2.52. The probe compares, every 50 steps of the mixed-team episode, the
actions of the current state with the actions the same state would get with every robot at
128 m, as |Δa| / |a| averaged over the robots of each group.)

Findings.

- **Dose-response, as-is map.** The cost rises with the fraction of long-range robots at
  every step (+0.37 ± 0.23 at f = 1/4, +0.79 at 1/2, +1.78 at 3/4 against f = 0, worse in
  11 of 11 environments each time) and the curve is convex: it lies below the straight line
  between f = 0 and f = 1 by 0.46 / 0.87 / 0.70 (10-11 of 11 environments at f ≤ 1/2). A
  quarter of long-range robots costs a fifth of the full collapse, half of them a quarter.
  So a few long-range robots do not drag the whole team down: the damage is mostly carried
  by the robots whose own inputs are out of distribution, and it is not a contagion that
  takes off as soon as one robot sees too far. With the map fixed at 256 m the curve is flat
  (1.71-2.01 at every f, differences from f = 0 within ±0.1 and 5-7 of 11 environments
  either way): the fix works for heterogeneous teams too, and a team where everyone hears
  far is even slightly better (1.71) than one where nobody does (1.93), since the graph
  carries more information and the map stays readable.
- **The graph does transmit the disturbance, measurably.** The frozen-state probe shows the
  128 m robots, whose own inputs are identical in both evaluations, changing their actions by
  18 to 35 % (relative) when a quarter to three quarters of their teammates hear far: the
  messages they receive through the TAGConv layers differ because their long-range neighbours'
  embeddings differ. The long-range robots themselves change by 116-128 %, more than their
  action magnitude. But the same probe under the fixed map still shows 12-31 % for the
  short-range robots (the long-range robots have 30 incoming edges instead of 2, so their
  embeddings differ even with a readable map) while the cost stays flat: a changed action is
  not a wrong action. The graph effect exists, is of the order of a quarter of the action,
  and is not what breaks the team; the broken map inputs of the long-range robots are.
- **Which robots hear far changes the cost by as much as the fraction does, but not in a way
  the start positions explain** (corrected on 18 October with 8 draws, section 13). The two
  draws of the long-range subset differ by 0.33 / 0.45 / 0.64 on average (up to 1.5) for
  f = 1/4, 1/2, 3/4 with the as-is map, as much as the whole effect of going from f = 0 to
  f = 1/4. With 8 draws the spread across draws within an environment is 0.30-0.54 (as-is)
  and 0.17-0.32 (fixed map), the order of the episode noise of section 5 (M3: 0.35 at 1024 m
  for a 1 m shift of the start), and none of four descriptors of where the long-range
  robots start predicts it (|Spearman ρ| ≤ 0.18, all p > 0.1). The first reading, "where the
  long-range robots stand decides how much they spoil", is withdrawn.

Figure `fig_hetero.png`; data `results_hetero/results_hetero.csv` (176 rows). Not done: a
model trained on heterogeneous teams, which would need the comm maps and edge weights of the
100k dataset rebuilt with per-robot ranges (only those two tensors depend on the range) and
one GPU run; and the question of transmission semantics (a link needing both robots in range,
or the sender's range) which would change the graph but not the maps.

## 13. Programme of 17-18 October (validated by Loïc, relayed by the planning session)

Submitted on 17 October, 21:30-23:30. Common criterion to report for every candidate: within
0.05 of V0 at 256 m and at the same level up to 1024 m, with no intervention at test time,
against the seed thresholds of section 8.

**Figure 20 of the paper** (one model per training range, each evaluated at its own range).
The 100k dataset is redrawn at 512 and 1024 m with `training/redraw_ranges.py`: only the
neighbour maps and the edge weights depend on the range (the clairvoyant trajectories, local
maps, obstacle maps and target actions do not). Checked against 200 samples generated natively
at 512 m: edge weights exact, maps within 1e-7 with the same nonzero cells, and the 128 m rule
applied to the same positions does not match. Redraw of 100k samples: 50 s at 512 m, 75 s at
1024 m (23 GB). Trainings r512 (40 epochs) and r1024 (25 epochs, since an epoch at 1024 m
costs about five times one at 128 m; 4 h 30 limits, best-validation checkpoint), jobs 1854545
and 1854548, then evaluations at the five ranges on both environment sets; the diagonal is
drawn by `lpac_study/plots_fig20.py` with the existing 128 m and 256 m models.

First half (18 October, 15:30). r512: 30 epochs in the 4 h 30 limit (453 s per epoch), best
validation 0.423 at epoch 5, then rising to 0.505; cost ratio 2.56 / 2.34 / 1.79 / 1.69 / 1.75
at 128-1024 m on the 11 release environments (2.57 / 2.34 / 1.85 / 1.77 / 1.92 on the 40 new
ones). At its own range it is 0.70 below the 128 m model (9 of 11, 36 of 40) and equal to the
256 m model (+0.05 ± 0.38, 7 of 11; +0.04, 21 of 40). It does not collapse at 1024 m (1.75,
the map scale is only doubled) and it is poor below its range (2.3-2.6 at 128-256 m, where
the map is zoomed in and nearly empty). Provisional diagonal 2.15 / 1.76 / 1.79 at 128 / 256 /
512 m; the 1024 m model is training (started 13:45).

**Complete (18 October, 20:00).** r1024: 17 epochs in 4 h 30 (842 s per epoch, 14 times
the 128 m cost: 30 neighbours per robot), best validation 0.512 at epoch 7 (validation
losses are not comparable across datasets, the maps differ), then 0.575. Cost ratio of
every model at every range (seed 0; r128 and r256 from sections 7-8), 11 release
environments, then 40 new ones:

| trained at | 128 m | 256 m | 512 m | 768 m | 1024 m |
|---|---|---|---|---|---|
| 128 m | **2.15** | 1.74 | 2.49 | 3.46 | 4.36 |
| 256 m | 2.28 | **1.76** | 1.74 | 1.90 | 2.31 |
| 512 m | 2.56 | 2.34 | **1.79** | 1.69 | 1.75 |
| 1024 m | 2.79 | 2.71 | 2.38 | 1.92 | **1.80** |
| 128 m (40 envs) | **2.13** | 1.82 | 2.53 | — | 4.76 |
| 256 m (40) | 2.42 | **1.85** | 1.81 | — | 2.29 |
| 512 m (40) | 2.57 | 2.34 | **1.85** | 1.77 | 1.92 |
| 1024 m (40) | 2.70 | 2.66 | 2.27 | 2.04 | **1.86** |

The diagonal (bold): 2.15 ± 0.51 / 1.76 ± 0.28 / 1.79 ± 0.33 / 1.80 ± 0.25 at 128 / 256 /
512 / 1024 m on the release set, 2.13 / 1.85 / 1.85 / 1.86 on the new set; centralized CVT
2.52 (2.33), decentralized 2.59 (2.44-2.54). Paired: the diagonal beats the centralized CVT
at every range, by 0.37 at 128 m (11 of 11) and 0.72-0.76 from 256 m (10 of 11) [0.20 to
0.48, 29-38 of 40]; 256 m is better than 128 m by 0.39 ± 0.38 (9 of 11) [0.28, 36 of 40];
512 m equals 256 m (+0.03 ± 0.17, 5 of 11) [0.00, 24 of 40] and 1024 m equals 512 m
(+0.01 ± 0.21, 4 of 11) [+0.01, 15 of 40]. At 1024 m the 1024 m model equals the 512 m one
(+0.05, 6 of 11; −0.05, 24 of 40) and is 2.6-2.9 below the 128 m one (11 of 11, 40 of 40).
Figures `fig_fig20.png` and `fig_fig20_new40.png`; data `results_fig20/`.

Does Fig. 20 reproduce? Partly. What reproduces: a model trained at its own range beats
both CVT baselines at every range, including 1024 m, so the collapse of sections 3-4 is a
property of the published 128 m checkpoint used out of its range, not of the architecture
at long range; and the improvement from 128 m to 256 m. What does not: the paper's curve
keeps improving up to 512 m and then declines moderately; ours is flat from 256 m on, to
within the seed noise (±0.02-0.08) and the environment noise (paired std 0.2), with
neither the rise at 512 m nor the decline at 1024 m. The off-diagonal is the mirror of
the collapse: every model is best near its training range, the long-range models are poor
below it (2.7-2.8 at 128-256 m for the 1024 m model, the map zoomed in and nearly empty),
and only the 128 m model collapses upward; the 512 m model is the most tolerant (1.7-1.8
from 512 to 1024 m).

Protocol differences that bound this comparison: one seed per range here, 11 and 40
environments and 600-step episodes against 100 environments in the paper; the 512 and
1024 m models were trained on the 128 m trajectories with the maps redrawn (identical to
native generation up to float rounding, section 13) and stopped by the 4 h 30 limit at 30
and 17 epochs with their best checkpoints at epochs 5 and 7, while the paper's number of
epochs per range is not stated; the paper's figure reports the cost with its own
normalization and no per-environment pairing. A difference of 0.05 on the diagonal is
below what this protocol can resolve.

**A. V0 trained with the fixed-scale map ("V0F").** The mixed-range dataset of section 11
with its maps redrawn at the fixed 256 m scale (robots within min(range, 256 m); the graph
keeps the real range), plus the control on the 128 m dataset alone; `redraw_ranges.py
--fixed-scale 256`, equal to the C3 drawing to the last bit. Trainings 1854568 (mixed, 25
epochs) and 1854569 (128 m, 40 epochs), evaluated with the fixed-scale map at inference
(their input convention) via the factorial level G0C3 at the five ranges, 11 and 40 envs.

Result of A (19 October, 00:00). Trainings: mixed set 25 epochs in 2 h 48, best validation
0.422 at epoch 11; 128 m set 40 epochs in 1 h 16, best validation 0.439 at epoch 12. Both
are evaluated with the fixed-scale map at inference, which is their input convention
(training = test, no mismatch); the map is drawn exactly as the inference-time fix of
section 9 draws it. Cost ratio (std across environments in brackets):

| | 128 m | 256 m | 512 m | 768 m | 1024 m |
|---|---|---|---|---|---|
| V0F mixed ranges, 11 release envs | 2.11 (0.50) | 1.86 (0.35) | **1.78** (0.35) | 1.76 (0.33) | 1.77 (0.33) |
| V0F 128 m, 11 envs | 2.06 (0.41) | 2.01 (0.41) | 2.11 (0.53) | 2.12 (0.56) | 2.14 (0.57) |
| V0 r128 s0, as is | 2.15 | **1.74** | 2.49 | 3.46 | 4.36 |
| V0 r128 s0, fixed map at inference only | 2.05 | = as is | 1.81 | 1.86 | 1.84 |
| V0 mixed ranges, as-is map (3 seeds) | 2.40 | 2.08 | 1.91 | 1.97 | 1.93 |
| V0F mixed ranges, 40 new envs | 2.14 (0.36) | 1.92 (0.38) | 1.92 (0.37) | 1.92 (0.37) | 1.92 (0.35) |
| V0F 128 m, 40 envs | 2.07 | 1.95 | 1.99 | 2.04 | 2.04 |
| V0 r128 s0, fixed map at inference only (40) | 2.06 | 1.82 | 1.78 | — | 1.80 |

Paired, 11 release environments [40 new ones]:

- V0F mixed − V0 r128 as is: +0.13 ± 0.20 at 256 m (3 of 11 better) [+0.10, 18 of 40];
  −0.71 / −1.69 / −2.59 at 512 / 768 / 1024 m (10-11 of 11) [−0.60, —, −2.84].
- V0F mixed − V0 r128 with the fixed map at inference: +0.06 / — / −0.02 / −0.10 / −0.07
  (4-7 of 11) [+0.07 / +0.10 / +0.15 / — / +0.12, 10-18 of 40].
- V0F mixed − V0 mixed ranges with the as-is map (seed 0): −0.29 / −0.20 / −0.12 / −0.21 /
  −0.11 (8-11 of 11) [−0.34 / −0.22 / −0.12 / −0.08 / −0.07, 25-35 of 40].
- V0F mixed − V0 trained at 256 m: +0.10 at 256 m (3 of 11), −0.54 at 1024 m (10 of 11).
- V0F mixed − centralized CVT: −0.42 to −0.76 at every range (10-11 of 11) [−0.19 to −0.41,
  29-35 of 40]. V0F mixed − V0F 128: −0.14 / −0.32 / −0.36 / −0.37 from 256 m (8-10 of 11).
- V0F 128 − V0 r128 with the fixed map at inference: +0.30 / +0.26 / +0.30 at 512-1024 m
  (2 of 11) [+0.22 / +0.24, 10 of 40]; +0.27 above V0 as is at 256 m.

Reading. Training with the fixed-scale map on mixed ranges gives a flat model (1.76-1.86
from 256 m, 1.92 on the new set) with no intervention at test time beyond its own input
convention: it equals the inference-time fix on the release set (differences within ±0.1,
4-7 of 11) and trails it by 0.1 on the new set, and it is the best of the trained recipes
across ranges (0.1-0.3 below the mixed-range model with as-is maps, 0.5 below the 256 m
model at 1024 m). Against the common criterion it misses the 256 m target: +0.13 above V0
on the release set and +0.10 on the new one, 3 of 11 and 18 of 40 environments better,
against a seed noise of 0.02 and a paired standard error of 0.06, so the gap is small but
real; from 512 m it is at V0's best level. The 128 m control fails: with the maps of the
128 m data drawn at the 256 m scale, the model never saw the robots between 128 and 256 m
that the fixed map shows at test time, and it ends 0.25-0.30 above the inference-time fix
beyond 512 m and 0.27 above V0 at 256 m. The fixed-scale map works when the training data
covers the range of neighbours the map can show.

**Three seeds of V0F on the mixed set** (19 October, 11:30; jobs 1857200 and 1857203, same
recipe, best validation 0.429 and 0.427 at epoch 7). Environment mean per seed and mean ±
std across the three seeds:

| V0F mixed ranges | 128 m | 256 m | 512 m | 768 m | 1024 m |
|---|---|---|---|---|---|
| seed 0 / 1 / 2, 11 release envs | 2.11 / 2.11 / 2.10 | 1.86 / 1.85 / 1.87 | 1.78 / 1.90 / 1.81 | 1.76 / 1.85 / 1.82 | 1.77 / 1.85 / 1.82 |
| mean ± std across seeds (11) | 2.11 ± 0.00 | 1.86 ± 0.01 | 1.83 ± 0.06 | 1.81 ± 0.04 | 1.82 ± 0.04 |
| mean ± std across seeds (40 new envs) | 2.09 ± 0.05 | 1.91 ± 0.02 | 1.91 ± 0.01 | 1.91 ± 0.04 | 1.92 ± 0.03 |
| V0 r128, mean ± std across seeds (11) | 2.15 ± 0.01 | 1.74 ± 0.02 | 2.46 ± 0.08 | 3.58 ± 0.14 | 4.61 ± 0.26 |

The criterion at 256 m, three seeds on each side: per-environment costs averaged over the
three seeds of each model, then paired across the 11 release environments, V0F − V0 =
+0.122 ± 0.156, standard error 0.047, V0F better in 2 of 11 environments; the nine
seed-pair differences of the environment mean lie between +0.09 and +0.15 (mean +0.122,
std 0.018). The gap is 0.12, two and a half standard errors above 0.05 and six seed
standard deviations away from zero: the criterion "within 0.05 of V0 at 256 m" is not met,
and the gap is a property of the recipe, not of a seed. From 512 m on, V0F (seed mean)
against V0 with the fixed map at inference (seed mean of the three r128 seeds): +0.06 /
−0.03 / −0.01 at 512 / 768 / 1024 m, standard errors 0.06, 5-7 of 11: equal. Against V0 as
is, −0.62 at 512 m (10 of 11) and −2.80 at 1024 m (11 of 11). On the 40 new environments
(three V0F seeds against the single V0 seed): +0.085 ± 0.211 at 256 m, standard error 0.033,
19 of 40, seed differences +0.06 to +0.10; −0.61 at 512 m (34 of 40), −2.84 at 1024 m (39 of
40); against the fixed map at inference +0.14 / +0.12 at 512 / 1024 m (standard error 0.03,
12 of 40): on the new set V0F trails the inference-time fix by 0.1.

Figure `fig_fixedmap_training.png` (V0F as the mean of three seeds); data `results_A/`
(factorial rows, level G0C3, one file per seed), loss histories `results_A/train_*/`.

**B. V3 trained on the mixed-range dataset** (job 1854570, 25 epochs), evaluated like the
other variants.

Result of B (19 October, 02:30). 25 epochs in 2 h 35, best validation 0.614 at epoch 20
(V0 on the same mixed set: 0.531; V3 on the 128 m set: 0.443), still falling slowly. Cost
ratio: 2.67 / 2.57 / 2.81 / 2.91 / 2.92 at 128-1024 m on the 11 release environments (std
0.5-0.7), 2.70 / 2.63 / 2.71 / 2.76 / 2.74 on the 40 new ones. Paired: +0.48 / +0.49 at
128 / 256 m against V3 trained at 128 m (0 of 11) and equal to it at 1024 m (+0.00);
+0.27 to +1.03 against V0 trained on the same mixed set (0-2 of 11); +0.56 to +1.15
against V0F (0 of 11); above the centralized CVT at every range (+0.04 to +0.40, 1-4 of
11) [40 envs: +0.30 to +0.42, 2-7 of 40]. Mixing the ranges in the training data makes V3
worse everywhere: its PointNet layer reads relative positions in units of 256 m, so the
1024 m samples feed it inputs four times larger than the 128 m ones, and one recipe for
both regimes fits neither (training loss 0.54 at the best epoch against 0.38 for V3 at
128 m). B fails the criterion by a wide margin; the fixed-scale map (A) is the recipe that
makes mixed-range training work, and it needs the map. Data `results_B/`, figure
`fig_mixed_range.png` (V3 mixed added).

**C. Heterogeneous teams, 8 draws per fraction** (section 12 had 2), both map modes, with
four descriptors of where the long-range robots start: distance to the centre, number of
robots within 128 m, Voronoi share of the importance field, spread. Job 1854555, CPU.

Result of C (18 October, 10:00; 572 episodes, 11 tasks of about 75 min). Cost ratio, mean
± std across environments of the environment mean over the 8 draws; in brackets the std
across draws within an environment and the max − min across draws:

| Fraction at 1024 m | 0 | 1/4 | 1/2 | 3/4 | 1 |
|---|---|---|---|---|---|
| as-is map | 1.98 ± 0.46 | 2.42 ± 0.46 [0.30, 0.93] | 2.89 ± 0.59 [0.38, 1.03] | 3.95 ± 0.97 [0.54, 1.59] | 5.30 ± 1.51 |
| map at a fixed 256 m scale | 1.93 ± 0.25 | 2.10 ± 0.19 [0.27, 0.76] | 1.98 ± 0.18 [0.32, 0.96] | 1.86 ± 0.14 [0.17, 0.50] | 1.71 ± 0.19 |
| probe, 128 m robots (as-is / fixed) | — | 0.21 / 0.14 | 0.29 / 0.22 | 0.36 / 0.31 | — |

- The 2-draw numbers of section 12 hold with 8 draws, slightly higher (2.42 / 2.89 / 3.95
  against 2.36 / 2.77 / 3.77): +0.44 / +0.91 / +1.97 against f = 0 (11 of 11 environments at
  every fraction), below the straight line between f = 0 and f = 1 by 0.39 / 0.75 / 0.52
  (10-11 of 11). The dose-response is convex. With the fixed map the curve is flat within
  ±0.2 (+0.17 at f = 1/4 is the largest deviation, 9 of 11; −0.07 at f = 3/4).
- **The draw matters, but not through where the long-range robots start.** Within an
  environment and a fraction, the 8 draws spread by 0.30-0.54 (as-is) and 0.17-0.32 (fixed
  map); the max − min reaches 1.6. Four descriptors of the subset at t = 0 (Voronoi share of
  the importance field, distance to the centre, number of robots within 128 m, spread) were
  correlated with the cost across draws, within (environment, fraction): every Spearman ρ is
  between −0.13 and +0.18 with p > 0.1, pooled over fractions |ρ| ≤ 0.09, and a linear model
  of the four together explains 2 % of the draw variance (as-is map). The descriptors do
  vary across draws (within-cell std 0.08 of the field share, 0.23 robots of local degree),
  so the test had something to work with. The spread across draws is therefore of the same
  nature as the episode noise measured in section 5 (M3: a 1 m shift of the start changes
  the 1024 m cost by 0.35): the collapsed regime is sensitive to any change of the initial
  condition, and the identity of the long-range robots is one more such change. Section 12's
  reading that the position of the long-range robots "decides how much they spoil" is
  withdrawn above.
- The probe values are unchanged (128 m robots change their actions by 21-36 % through the
  graph with the as-is map, 14-31 % with the fixed one, where it costs nothing).

Figures `fig_hetero.png` (now 8 draws) and `fig_hetero_position.png` (cost against each
descriptor, both centred within (environment, fraction)); data
`results_hetero/results_hetero8.csv`.

**H1. Mixture of specialists at inference** (`lpac_study/mixture.py`): in the 128/1024 m
team, each robot runs the model trained at its own range; the two models are applied layer
by layer on the mixed inputs and each robot keeps its own model's output (CNN, each TAGConv
layer, action MLP, de-normalization), so the messages a robot aggregates were produced by
its neighbours' models. Modes: mixture, short specialist on everyone, long specialist on
everyone; checked locally that the single-specialist modes reproduce the specialists.
Hypothesis: two specialists do not share a message "language", so the mixture may be worse
than either alone. Waits for the 1024 m model of the figure-20 chain.

Result of H1 (18 October, 20:30; 440 episodes, job 1856368, specialists r128 seed 0 and the
r1024 of the figure-20 chain, 4 draws of the long-range subset, as-is maps). Cost ratio,
mean ± std across environments of the environment mean over draws:

| Fraction at 1024 m | 0 | 1/4 | 1/2 | 3/4 | 1 |
|---|---|---|---|---|---|
| 128 m specialist on every robot | 2.15 ± 0.51 | 2.35 ± 0.40 | 2.73 ± 0.62 | 3.51 ± 0.74 | 4.36 ± 1.08 |
| 1024 m specialist on every robot | 2.79 ± 0.62 | 2.55 ± 0.41 | 2.31 ± 0.27 | 2.02 ± 0.26 | 1.80 ± 0.25 |
| mixture (each robot its own specialist) | = 128 m | **2.17 ± 0.41** | **2.24 ± 0.44** | **1.95 ± 0.19** | = 1024 m |
| published model, as-is map (section 13, 8 draws) | 1.98 | 2.42 | 2.89 | 3.95 | 5.30 |
| published model, map at a fixed 256 m scale | 1.93 | 2.10 | 1.98 | 1.86 | 1.71 |

Paired within environments: mixture − 128 m specialist −0.17 ± 0.11 (11 of 11) at f = 1/4,
−0.49 (10 of 11) at 1/2, −1.55 (11 of 11) at 3/4; mixture − 1024 m specialist −0.37 ± 0.21
(11 of 11), −0.07 ± 0.39 (7 of 11), −0.07 ± 0.15 (7 of 11); mixture − best specialist of the
two −0.16 (11 of 11), −0.01 (6 of 11), −0.07 (7 of 11); mixture − published model −0.25
(8 of 11), −0.66 (10 of 11), −2.00 (11 of 11); mixture − fixed-scale map +0.07 (5 of 11),
+0.25 (3 of 11), +0.10 (4 of 11). The specialists out of place: the 1024 m model on an
all-128 m team costs +0.64 (11 of 11), the 128 m model on an all-1024 m team +2.56 (11 of 11).

First reading (superseded by the control below): the mixture is never worse than the better
specialist and beats both at f = 1/4 in every environment, so the two specialists would not
need a common message language.

**Control: specialists from different training seeds** (requested by the planning session;
18 October, 22:00). The two specialists above share seed 0: the same initial weights and the
same sample order (r1024 was trained on the 128 m trajectories with the maps redrawn), so
their compatibility could come from that common origin. The grid was rerun with the 128 m
specialist of seed 1 and of seed 2 against the same r1024 (jobs 1856566-1856567, same 4
draws, as-is maps). Mixture, mean across environments at f = 1/4 / 1/2 / 3/4:

| 128 m specialist | mixture | 128 m model alone at the same f | mixture − best specialist (n mixture better) |
|---|---|---|---|
| seed 0 (common with r1024) | 2.17 / 2.24 / 1.95 | 2.35 / 2.73 / 3.51 | −0.16 (11/11) / −0.01 (6/11) / −0.07 (7/11) |
| seed 1 | 2.41 / 2.36 / 2.25 | 2.39 / 2.82 / 3.52 | +0.09 (4/11) / +0.09 (4/11) / +0.23 (2/11) |
| seed 2 | 2.30 / 2.25 / 2.12 | 2.43 / 2.73 / 3.47 | −0.05 (8/11) / −0.01 (7/11) / +0.10 (3/11) |

(1024 m model alone: 2.55 / 2.31 / 2.02.) Paired against the common-seed mixture: seed 1
+0.24 ± 0.38 / +0.12 ± 0.56 / +0.30 ± 0.32, seed 2 +0.12 ± 0.29 / +0.01 ± 0.46 / +0.17 ± 0.26.
Against the fixed-scale map: +0.20 to +0.39 for seeds 1-2 (2-4 of 11) instead of +0.07 to
+0.25 for seed 0.

Representations (`lpac_study/cka.py`, linear CKA between the two models' layer outputs on
the same 7 states of a heterogeneous episode, environments 0 and 3, f = 1/4 and 1/2): the
common-seed pair is no closer than the different-seed pairs. At the first TAGConv layer
0.52 / 0.57 / 0.58 (seed 0 / 1 / 2 against r1024, env 0, f = 1/2), at the fifth 0.30 / 0.34
/ 0.34, on the actions 0.01 / 0.01 / 0.00; for the short-range robots alone 0.68-0.79 at the
first layers in all three cases. Reference: two 128 m seeds against each other 0.8-0.9 at
every layer; r128 against r256 0.3-0.7. The two specialists' representations differ a lot,
equally for a common or a different seed.

Reformulated reading. The common seed does help the mixture, by 0.1-0.3 (consistent in sign
at every fraction, at the edge of significance with 11 environments), even though linear CKA
does not see a closer representation. With specialists from independent seeds the mixture
is at the level of the better specialist used alone, not above it: −0.05 to +0.23, 2-8 of 11
environments better, inside the draw and environment noise (0.2-0.4). What survives the
control: the mixture is never much worse than the better specialist (worst case +0.23 at
f = 3/4, where the 128 m specialist alone is 1.3 worse), so there is no sign of two
incompatible message languages making the mixed team fail; what does not survive: the
claim that the mixture beats both specialists. The fixed-scale map on the published model
remains better than every mixture by 0.1-0.4, with one model and no switching.

Figure `fig_mixture.png` (mixtures of the three seeds); data `results_h1/results_h1*.csv`.

**H2. One model for heterogeneous teams, range as input ("V0R").** The 100k dataset redrawn
with a range drawn per robot in {128, 256, 512, 1024} m (reception semantics: robot i hears
the robots within r_i, edge j → i, map at the scale r_i; `redraw_ranges.py --hetero`, seed 0,
12.9 incoming edges per robot on average), and the V0 architecture with r_i / 1024 appended
to the 34 GNN inputs (`LPACRangeInput`, 1,807,234 parameters, +1,024). Checked locally:
edge orientation (in-degree = robots within own range, graph directed), maps and edges equal
to the inference-time construction of section 12, weights = exp(−d²/r_target²); training
step and both policies run. Training queued behind B (40 epochs), then the homogeneous
sweeps (11/40 envs) and the heterogeneous grid f ∈ {0, 1/4, 1/2, 3/4, 1}.

Result of H2 (19 October, 09:00). 40 epochs in 4 h 19, best validation 0.513 at epoch 9,
then rising to 0.586 (validation set with per-robot ranges, not comparable with the others).
Cost ratio (std across environments in brackets):

| homogeneous team at | 128 m | 256 m | 512 m | 768 m | 1024 m |
|---|---|---|---|---|---|
| V0R, 11 release envs | 2.75 (0.71) | 2.16 (0.39) | 1.87 (0.27) | 2.05 (0.51) | 1.89 (0.27) |
| V0R, 40 new envs | 2.61 (0.40) | 2.17 (0.37) | 1.91 (0.34) | 1.98 (0.29) | 1.91 (0.29) |
| V0 r128, as is (11) | 2.15 | **1.74** | 2.49 | 3.46 | 4.36 |
| V0 r128, fixed map at inference (11) | 2.05 | = | 1.81 | 1.86 | 1.84 |
| V0F mixed, A (11) | 2.11 | 1.86 | 1.78 | 1.76 | 1.77 |
| V0 mixed ranges, as-is map (11, seed 0) | 2.40 | 2.06 | 1.90 | 1.97 | 1.89 |

| heterogeneous team, fraction at 1024 m | 0 | 1/4 | 1/2 | 3/4 | 1 |
|---|---|---|---|---|---|
| V0R (one model, its own convention) | 2.75 (0.71) | 2.49 (0.54) | 2.16 (0.25) | 2.08 (0.28) | 1.89 (0.27) |
| published model, as-is map (8 draws) | 1.98 | 2.42 | 2.89 | 3.95 | 5.30 |
| published model, fixed-scale map | 1.93 | 2.10 | 1.98 | 1.86 | 1.71 |
| mixture of specialists (H1, seed 0) | 2.15 | 2.17 | 2.24 | 1.95 | 1.80 |

Paired, 11 release environments [40 new ones]. Homogeneous: V0R − V0 as is +0.60 ± 0.45 at
128 m (0 of 11) [+0.48, 0 of 40], +0.42 ± 0.40 at 256 m (1 of 11) [+0.35, 5 of 40], −0.6 /
−1.4 / −2.5 at 512 / 768 / 1024 m (10-11 of 11); V0R − fixed map at inference +0.71 at
128 m, then +0.07 / +0.19 / +0.05 (4-6 of 11) [+0.14, +0.11, 12-14 of 40]; V0R − V0F
+0.65 / +0.30 / +0.09 / +0.29 / +0.12 (0-5 of 11) [+0.48 / +0.25 / −0.01 / +0.05 / −0.01];
V0R − V0 mixed ranges with as-is maps +0.35 / +0.10 / −0.03 / +0.07 / +0.01 (2-6 of 11)
[+0.14 / +0.03 / −0.13 / −0.02 / −0.08]; V0R − centralized CVT +0.23 at 128 m (3 of 11),
−0.36 to −0.65 from 256 m (8-9 of 11). Heterogeneous: V0R − published as is +0.77 at f = 0
(0 of 11), +0.06 at 1/4, −0.73 / −1.87 / −3.41 from 1/2 (10-11 of 11); V0R − published with
the fixed map +0.82 / +0.38 / +0.18 / +0.22 / +0.19 (0-4 of 11); V0R − mixture of
specialists +0.60 / +0.31 / −0.07 / +0.13 / +0.09 (0-7 of 11). Frozen-state probe under V0R:
the 128 m robots change their actions by 16 / 34 / 54 % at f = 1/4 / 1/2 / 3/4 (published
model: 21 / 29 / 36 %).

Reading. Telling the model its own range, and training it on teams where every robot has
its own, removes the collapse (1.9-2.1 from 512 m, 1.9 at f = 1 and at 1024 m) but buys
nothing over the plain mixed-range recipe: at every range the two are equal within ±0.1
(2-6 of 11 environments either way, on both sets). The range input is redundant with what
the map already shows the CNN (a map drawn at r_i looks different at each r_i). Where V0R
loses is the short range: 2.75 at 128 m and 2.16 at 256 m against 2.15 and 1.74 for the
128 m model, 0.4-0.6 above every other recipe, in 0-1 of 11 environments better. In its
training set a quarter of the robot-samples are at 128 m, against all of them for r128, and
the 128 m regime (sparse graph, nearly empty map) is the one that needs the most data. In
heterogeneous teams it is a single model that handles any mix without switching, better than
the published model as soon as half the team hears far, but 0.2-0.4 above the published
model with the fixed-scale map at every fraction and 0.1-0.3 above the mixture of
specialists except at f = 1/2. Criterion: failed at 256 m by 0.42; flat from 512 m.

Figure `fig_range_input.png` (homogeneous and heterogeneous panels); data `results_H2/`
(homogeneous sweeps on both sets, heterogeneous grid, loss history).

**H3** (shared GNN, one CNN per range) is not written: it waits for H1 and H2 and for an
explicit go. V4 (bounded edge attributes) is Loïc's.

Execution order on the GPU partition (one job per user accrues queue age, so the trainings
run one after the other in submission order): r512 → r1024 → A mixed → A 128 → B → H2.
CPU in parallel: C now, H1 after r1024.

## 14. V4 and V5: towards a model that improves with the range (19 October, launched 14:30)

Approved by Loïc, relayed by the planning session. The open question after section 13: every
recipe so far is at best flat beyond 256 m; is there one that gets *better* with the range,
as the paper's Fig. 20 suggests?

**V4, V3 with bounded edge features.** Same as V3 (no neighbour map, residual PointNet-style
front layer, unchanged TAGConv stack, same widths), but the edge input is no longer
(p_j − p_i) / 256 m, which grows with the range and is why V3 fails on mixed ranges
(section 13, B): per edge j → i with d = |p_j − p_i|, e_ij = [(p_j − p_i) / d, tanh(d / 256 m)],
three numbers in [−1, 1]; self-loop e = 0. `BoundedGeometryConv` (a small MessagePassing
layer: message = local_nn([x_j, e_ij]), max aggregation, global_nn, self-loops) and
`LPACBoundedGeometry` in `lpac_study/variants.py`; 1,889,892 parameters (V3 + 256), 16 hops.
V3 against V4 isolates the edge encoding.

**V5, multi-scale fixed neighbour maps.** The original LPAC with three pairs of neighbour
channels drawn at fixed scales 256, 512 and 1024 m (16 / 32 / 64 m per cell on the same
32 × 32 grid, value (p_j − p_i) / scale), each pair holding the robots within min(range,
scale): CNN input 8 channels, nothing else changes; 1,807,362 parameters. At 128 m the three
pairs show the same neighbours at three zooms; at 1024 m the 256 m pair is the fixed-scale
map of section 9 and the 1024 m pair is the original map. Datasets redrawn from the stored
positions (`redraw_ranges.py --multiscale`, with `--infer-range` recovering each sample's
range of the merged mixed set from its edge weights); the six channels are kept sparse until
a sample is requested (`MultiScaleDataset`), 78 GB dense otherwise.

Unit tests (`tests/test_variants_v4v5.py`, 6 passing locally and on FarmShare): V4 features
bounded and invariant to a scaling of the positions except through the tanh; at 128 m every
V5 pair is the drawing of exactly the in-range neighbours; at 1024 m the 256 m pair equals
the C3 map bit for bit; the dataset and inference builders agree.

Runs (one L40S each, best-validation checkpoint, evaluations at 128-1024 m on the 11 release
and the 40 new environments): trials of 3 epochs on 2,500 samples (jobs 1858441-1858442),
then in priority order V5 on mixed ranges (3 seeds, 25 epochs, jobs 1858443-1858449), V4 on
mixed ranges (3 seeds, 25 epochs, 1858452-1858458), V4 at 128 m (seed 0, 100 epochs,
1858461), V5 at 128 m (seed 0, 40 epochs, sanity check, 1858464). Estimated 22 GPU-hours;
V4 at 128 m seeds 1-2 (4-5 more) wait for an explicit go. To report per variant and seed:
cost against range next to the published model, V0F (A) and V3; (a) within 0.05 of V0 at
256 m? (b) is the cost at 1024 m significantly lower than at 256 m (paired, seed noise
0.02-0.05 at 256 m)? and the final training loss against V0 / V3 to separate underfitting
from transfer failure.

**Incidents.** The first dataset build failed on five of the 100,000 merged samples whose
range could not be inferred exactly (a pair at d close to the range whose stored float32
weight falls on the other side of the 1/e threshold); the inference now takes the closest
candidate when at most four pairs differ. The Slurm chain started V4 mixed seed 1 before
the trials; the run is valid (same data as the others) and was kept. The three V5 mixed
runs hit the 4 h 30 limit after 8-9 of 25 epochs: 1,640 s per epoch instead of the ~400 s
of V0F on the same data, because indexing a coalesced sparse tensor by sample scans all of
its 190 M entries (0.3 s per sample, 20 min per epoch across 27 workers); the loader now
splits the entries per sample once at load time (`MultiScaleDataset.comm_map`, same maps,
0.1 ms per sample on the small set). Measured on the real mixed training set (70,000
samples, 117 M nonzero entries, `training/time_loader.py`, job 1861155): 180 ms per sample
with whole-tensor indexing against 0.5 ms with the per-sample split, identical maps; the
split itself takes 267 s once at load time. Their evaluations below are labelled partial and
superseded: the three seeds were restarted from scratch with the fixed loader on 20 October
(approved by Loïc, about 9.5 GPU-hours), behind the V4 128 m seeds.

**Results (20 October, 16:00).** Training (epochs, best validation and its epoch, training
loss at the best epoch, final training loss): V4 mixed 25 / 0.351-0.356 at 21-25 / 0.29 /
0.28-0.29 (three seeds; V0 on the same mixed set 0.531 and V0F 0.422 at their best, V3 mixed
0.614); V4 at 128 m 100 / 0.408 at 24 / 0.37 / 0.212 (V0 0.449 and 0.21, V3 0.443 and 0.20);
V5 at 128 m 40 / 0.440 at 11 / 0.41 / 0.298; V5 mixed, partial, 8-9 / 0.465-0.470 at 4-6 /
0.36-0.42 / 0.30-0.32. V4 is not under-fitted anywhere: on the mixed set its validation loss
is the lowest of every mixed-range model by a wide margin (0.35 against 0.42 for V0F), so
whatever it does at test time is transfer, not training.

Cost ratio, environment mean, mean ± std across seeds where there are three:

| | 128 m | 256 m | 512 m | 768 m | 1024 m |
|---|---|---|---|---|---|
| V4 mixed ranges, 3 seeds, 11 release envs | 2.21 ± 0.01 | 1.88 ± 0.05 | 1.77 ± 0.08 | 1.76 ± 0.06 | **1.76 ± 0.06** |
| V4 mixed ranges, 3 seeds, 40 new envs | 2.14 ± 0.04 | 1.87 ± 0.03 | 1.81 ± 0.02 | 1.81 ± 0.01 | 1.82 ± 0.02 |
| V4 at 128 m, seed 0 (11 / 40) | 2.15 / 2.09 | 1.85 / 1.88 | 1.91 / 1.91 | 1.93 / 1.92 | 1.92 / 1.93 |
| V5 at 128 m, seed 0 (11 / 40) | 1.99 / 2.09 | **1.77** / 1.88 | 1.87 / 2.05 | 2.03 / 2.05 | 1.97 / 2.16 |
| V5 mixed, partial 8-9 epochs, 3 seeds (11) | 2.13 ± 0.03 | 1.90 ± 0.05 | 1.83 ± 0.06 | 1.85 ± 0.07 | 1.86 ± 0.08 |
| V0 r128, 3 seeds (section 10) | 2.15 ± 0.01 | **1.74 ± 0.02** | 2.46 ± 0.08 | 3.58 ± 0.14 | 4.61 ± 0.26 |
| V3 at 128 m, 3 seeds (section 11) | 2.24 ± 0.06 | 2.01 ± 0.06 | 2.31 ± 0.16 | 2.49 ± 0.22 | 2.58 ± 0.29 |
| V0F mixed, 3 seeds (section 13) | 2.11 ± 0.00 | 1.86 ± 0.01 | 1.83 ± 0.06 | 1.81 ± 0.04 | 1.82 ± 0.04 |
| published (section 3) | 1.98 | 1.69 | 2.77 | 4.52 | 5.22 |

Question (a), within 0.05 of V0 at 256 m (per-environment costs averaged over seeds, paired
against the three V0 seeds, 11 release envs [40 new ones against the single V0 seed]):
V4 mixed +0.139 ± 0.136, standard error 0.041, 2 of 11 better: no [+0.052, SE 0.031, 18 of
40: just outside]; V4 at 128 m +0.111, SE 0.074, 3 of 11: no [+0.063, 20 of 40]; V5 at 128 m
+0.032 ± 0.152, SE 0.046, 5 of 11: **yes** [+0.060, SE 0.043, 15 of 40: borderline]; V5
mixed partial +0.158: no.

Question (b), cost at 1024 m minus cost at 256 m, paired per environment: V4 mixed
−0.122 ± 0.097, SE 0.029, 10 of 11 environments lower at 1024 m [−0.055, SE 0.016, 27 of
40]: **yes, significantly lower**, the first model of this study whose cost keeps falling
beyond 256 m. V0F for reference −0.045, SE 0.035, 9 of 11 (not significant) [+0.012]; V5
mixed partial −0.040, SE 0.032 (not significant); V4 at 128 m +0.072 and V5 at 128 m
+0.200 (worse at 1024 m).

Paired comparisons, seed 0, 11 release envs [40 new ones]:

- V4 mixed − V3 at 128 m: +0.04 at 128 m, −0.14 at 256 m (8 of 11), −0.64 / −0.91 / −1.09 at
  512 / 768 / 1024 m (9-11 of 11) [−0.23 to −1.00 from 256 m, 34-39 of 40].
- V4 mixed − V0F mixed: +0.12 / +0.07 / +0.07 / +0.07 / +0.06 (2-5 of 11) [−0.02 / −0.03 /
  −0.10 / −0.09 / −0.08, 20-28 of 40]: equal on the release set, 0.1 better on the new set.
- V4 mixed − V0 mixed with as-is maps: −0.18 / −0.13 / −0.04 / −0.14 / −0.06 (6-9 of 11)
  [−0.36 / −0.25 / −0.22 / −0.17 / −0.15, 30-37 of 40].
- V4 mixed − centralized CVT: −0.30 / −0.59 / −0.67 / −0.69 / −0.69 (7-11 of 11).
- V4 at 128 m − V3 at 128 m: −0.04 / −0.22 / −0.59 / −0.81 / −1.00 (7-10 of 11) [−0.03 /
  −0.23 / −0.62 / −0.84 / −0.91, 23-40 of 40]: V4 beats V3 by far more than the seed noise
  of V3 (0.06-0.29), so seeds 1 and 2 of V4 at 128 m were launched under the agreed rule.
- V4 at 128 m − V0 r128: −0.00 / +0.11 / −0.58 / −1.53 / −2.44: trained at 128 m only, V4 does
  not collapse at all (1.9 at every range from 512 m).
- V5 at 128 m − V0 r128: −0.16 / +0.04 / −0.62 / −1.43 / −2.39 (5-11 of 11); V5 at 128 m −
  V0F mixed: −0.11 / −0.09 / +0.09 / +0.27 / +0.20.

Reading. V4 answers the open question in part. The bounded edge encoding does what V3's
raw relative position could not: trained on mixed ranges, V4 fits the data far better than
any other mixed-range model (validation 0.35), stays below the centralized CVT everywhere,
equals V0F on the release set and beats it by 0.1 on the new set, and its cost keeps
decreasing from 256 to 1024 m (−0.12, 10 of 11 environments), which no other model does.
It still pays 0.14 at 256 m against the 128 m model, like every mixed-range recipe, so
(a) is not met. Trained at 128 m alone, V4 does not collapse either (1.9 at every range),
which V3 only avoided after 100 epochs and only down to 2.6: the edge encoding, not the
training range, is what makes the geometry transferable. V5 at 128 m is the one model that
meets (a): three fixed-scale map pairs give the 256 m optimum of V0 (1.77) and no collapse
(1.97 at 1024 m) from the 128 m data alone, but it does not improve with the range (+0.20).
V5 on mixed ranges is unresolved (8-9 epochs); what it shows so far is V0F-like.

**Completed seeds (21 October).** V4 at 128 m seeds 1-2 (100 epochs, best validation 0.411 and
0.415 at epochs 26 and 20, final training loss 0.22 and 0.21) and V5 on mixed ranges restarted
from scratch with the fixed loader (25 epochs in 2 h 30 each, 315 s per epoch; best validation
0.460 / 0.468 / 0.454 at epochs 5 / 4 / 5, final training loss 0.19 / 0.20 / 0.18; the partial
runs above are superseded).

| | 128 m | 256 m | 512 m | 768 m | 1024 m |
|---|---|---|---|---|---|
| V4 at 128 m, 3 seeds, 11 release envs | 2.10 ± 0.04 | 1.88 ± 0.08 | 1.91 ± 0.02 | 1.96 ± 0.08 | 1.98 ± 0.08 |
| V4 at 128 m, 3 seeds, 40 new envs | 2.05 ± 0.03 | 1.87 ± 0.02 | 1.90 ± 0.01 | 1.90 ± 0.02 | 1.90 ± 0.02 |
| V5 mixed ranges, 3 seeds, 11 release envs | 2.12 ± 0.03 | **1.78 ± 0.04** | 1.80 ± 0.05 | 1.82 ± 0.02 | 1.82 ± 0.03 |
| V5 mixed ranges, 3 seeds, 40 new envs | 2.07 ± 0.01 | 1.88 ± 0.02 | 1.89 ± 0.03 | 1.90 ± 0.03 | 1.90 ± 0.03 |
| V4 mixed ranges, 3 seeds (above), 11 / 40 | 2.21 / 2.14 | 1.88 / 1.87 | 1.77 / 1.81 | 1.76 / 1.81 | **1.76** / 1.82 |

Question (a) at 256 m, seed-averaged and paired, 11 release envs [40 new ones]: V4 at 128 m
+0.140 ± 0.182, SE 0.055, 3 of 11: no [+0.045, SE 0.033, 19 of 40: yes]; V5 mixed
+0.037 ± 0.114, SE 0.034, 4 of 11: **yes** [+0.054, SE 0.033, 17 of 40: just outside]; V4
mixed +0.139: no [+0.052]. Question (b), 1024 m minus 256 m: V4 at 128 m +0.100 (SE 0.054,
3 of 11 lower at 1024 m) [+0.035]: no; V5 mixed +0.045 (SE 0.059, 6 of 11) [+0.027]: flat,
no; V4 mixed −0.122 (SE 0.029, 10 of 11) [−0.055, SE 0.016, 27 of 40]: yes, still the only one.

Paired, seed-averaged, 11 release envs [40 new ones]:

- V5 mixed − V4 mixed: −0.09 at 128 m (8 of 11), −0.10 ± 0.11 at 256 m (SE 0.03, 10 of 11),
  +0.03 / +0.06 / +0.07 at 512-1024 m (5-8 of 11) [−0.07, 0.00, +0.08 / +0.08 / +0.08,
  14-16 of 40]: V5 is the better model at 256 m, V4 at 768-1024 m, by 0.1 either way.
- V5 mixed − V0F: +0.01 / −0.08 (8 of 11) / −0.03 / +0.01 / +0.01 [−0.02 to −0.03, 22-27 of
  40]: the three fixed-scale pairs improve on the single fixed-scale map by 0.08 at 256 m
  and are equal elsewhere.
- V5 mixed − V0 r128: −0.03 / +0.04 / −0.65 / −1.76 / −2.79 (7-11 of 11); V5 mixed −
  centralized CVT −0.40 to −0.75 (10-11 of 11).
- V4 at 128 m − V3 at 128 m: −0.14 / −0.13 / −0.40 / −0.53 / −0.60 (10-11 of 11) [−0.07 to
  −0.53, 27-40 of 40]: the bounded encoding beats V3 at every range by 2-6 standard errors.
- V4 at 128 m − V4 mixed: −0.11 at 128 m (10 of 11), 0.00 at 256 m, +0.13 / +0.20 / +0.22
  at 512-1024 m (0-1 of 11): training on mixed ranges buys V4 0.2 at long range and costs
  0.1 at 128 m.

Reading. Two different answers to the two questions. V5 on mixed ranges is the model that
keeps the 256 m optimum: 1.78, within 0.04 of V0, with a flat 1.80-1.82 up to 1024 m, below
the centralized CVT everywhere and 0.08 better than V0F at 256 m; it is the first trained
recipe that meets criterion (a) on the release set (just outside on the new set, +0.054),
and it does so with the original architecture plus four map channels. V4 on mixed ranges is
the model that improves with the range: 1.88 at 256 m but 1.76 at 1024 m (−0.12, 10 of 11
environments), 0.07 below V5 there. Neither does both. V4 at 128 m alone, with three seeds,
confirms that the bounded edge encoding removes the collapse by itself (1.9-2.0 at every
range) and beats V3 at every range, but it does not improve with the range (+0.10) and sits
0.14 above V0 at 256 m. By the agreed order, (b) first and (a) second, V4 mixed is the
candidate taken to the heterogeneous-team evaluation (jobs below); V5 mixed is the better
candidate if the 256 m level is what matters.

**Check requested by the planning session: why do the partial and the full V5-mixed runs
differ at 256 m (1.90 ± 0.05 against 1.78 ± 0.04) when their selected checkpoints are equally
early?** (1) Both sets of evaluations used `model_best_state_dict.pt`. (2) Training is not
deterministic at a fixed seed: `train_launch.py` seeds torch and CUDA, but the GPU
scatter operations of TAGConv and the 27 loader workers are not bit-reproducible, and the
weights of the two launches of the same seed already differ by 18 % (relative norm) after
the first epoch and by 40 % at the selected checkpoint; the validation curves are close but
not identical (seed 0: 0.465 at epoch 6 against 0.460 at epoch 5). The partial runs are
therefore not "under-trained" in the sense that matters: every V5-mixed run reaches its best
validation at epoch 4-6 and overfits afterwards (training loss 0.19 at epoch 25, validation
up to 0.54), so "25 epochs" is early stopping at epoch 5, and the partial launch stopped at
8-9 epochs selected the same kind of checkpoint. The two launches are six legitimate runs of
the recipe. (3) At the same seed the two launches differ at 256 m by 0.08 / 0.15 / 0.13
(release set), more than the seed-to-seed spread within a launch (0.05 and 0.04): the
three-seed standard deviations quoted throughout this journal underestimate the run-to-run
noise of a recipe, at least on the 11 release environments (on the 40 new ones the same
pairs differ by 0.05 / 0.04 / 0.01, and the seed spread is 0.02-0.04).

Pooling the six runs: V5 mixed 2.13 ± 0.03 / 1.84 ± 0.08 / 1.82 ± 0.05 / 1.83 ± 0.05 /
1.84 ± 0.06 at 128-1024 m on the release set, 2.09 / 1.89 / 1.89 / 1.90 / 1.90 on the new
one. Criterion (a) with the six-run average, paired against V0: +0.097 ± 0.134, standard
error 0.040 (0.032 across runs), 2 of 11 [+0.068, SE 0.033, 18 of 40]. **The (a) claim for V5
mixed is withdrawn**: with the honest noise it is 0.10 above V0 at 256 m, like V0F
(+0.12) and V4 mixed (+0.14); the three full runs were a favourable draw. Criterion (b) is
unchanged (flat: +0.003, SE 0.038). What stands: V5 mixed is flat from 256 m at 1.82-1.84,
0.02-0.08 below V0F at 256 m (within the run noise), below the centralized CVT everywhere;
V4 mixed remains the only model whose cost falls with the range, and the hetero evaluation
on it is unaffected. The same caveat applies to every "within 0.05" statement of sections
13-14 made with three seeds: the relevant noise at 256 m on the release set is about 0.08
per run, so a 0.05 margin is not resolvable without 6-10 runs.

**Heterogeneous teams with V4 mixed (21 October, 16:30; job 1864176, seed 0, 11 release
environments, f ∈ {0, 1/4, 1/2, 3/4, 1}, 4 draws of the long-range subset, no map to draw).**
Cost ratio, mean ± std across environments:

| Fraction at 1024 m | 0 | 1/4 | 1/2 | 3/4 | 1 |
|---|---|---|---|---|---|
| V4 mixed, one model for everyone | 2.22 ± 0.35 | 2.20 ± 0.33 | 2.07 ± 0.27 | 1.96 ± 0.29 | **1.83** ± 0.29 |
| published model, as-is map (8 draws) | 1.98 | 2.42 | 2.89 | 3.95 | 5.30 |
| published model, fixed 256 m map | 1.93 | 2.10 | 1.98 | 1.86 | 1.71 |
| V0R, own range as input (section 13) | 2.75 | 2.49 | 2.16 | 2.08 | 1.89 |
| mixture of the two specialists (H1) | 2.15 | 2.17 | 2.24 | 1.95 | 1.80 |

Paired: V4 mixed − published as is +0.24 at f = 0 (3 of 11), −0.23 at 1/4 (9 of 11), −0.83 /
−1.99 / −3.47 from 1/2 (10-11 of 11); V4 mixed − published with the fixed map +0.30 at f = 0
(2 of 11), then +0.09 / +0.08 / +0.10 / +0.12 (2-6 of 11); V4 mixed − V0R −0.53 / −0.29 /
−0.10 / −0.12 / −0.06 (7-11 of 11); V4 mixed − mixture of specialists +0.07 / +0.02 / −0.17 /
+0.01 / +0.03 (3-8 of 11). From f = 0 to f = 1 the cost falls by 0.39 ± 0.20 (11 of 11), and
it falls at every step (2.22 → 2.20 → 2.07 → 1.96 → 1.83). Frozen-state probe: the 128 m
robots change their actions by 19 / 34 / 43 % at f = 1/4 / 1/2 / 3/4 (published model 21 /
29 / 36 %), the 1024 m robots by 150 / 113 / 110 %.

Reading. V4 mixed is the first model for which every robot that hears farther helps the
team: the cost decreases monotonically with the fraction of long-range robots, in every
environment, which neither the published model (which collapses), nor the fixed-scale map
(flat, with a dip at f = 1/4), nor V0R (which starts far too high), nor the mixture (not
monotone) does. It beats V0R at every fraction, equals the mixture of specialists with a
single model and no switching, and loses only to the inference-time fix on the published
model, by 0.1 at every fraction above zero and by 0.3 at f = 0, where its own weakness at
128 m (2.2 against 2.0) shows. In a heterogeneous team it is the best single trained model
of this study.

Figure `fig_hetero_candidates.png`; data `results_V45/results_hetero_V4mixed.csv`.

**Heterogeneous teams, three seeds of V4 mixed (22 October, 00:30; jobs 1865568-1865569).**
Per seed, mean across environments at f = 0 / 1/4 / 1/2 / 3/4 / 1: seed 0 2.22 / 2.20 / 2.07 /
1.96 / 1.83 (drop from f = 0 to 1 −0.39 ± 0.20, 11 of 11); seed 1 2.21 / 2.20 / 2.14 / 1.98 /
1.72 (−0.49 ± 0.26, 11 of 11); seed 2 2.20 / 2.10 / 2.01 / 1.85 / 1.72 (−0.48 ± 0.34, 10 of 11).
Mean ± std across seeds 2.21 ± 0.01 / 2.17 ± 0.06 / 2.07 ± 0.07 / 1.93 ± 0.07 / 1.76 ± 0.06; the
environment means decrease at every step for every seed (within environments the sequence is
monotone in 3 / 5 / 6 of 11, the rest showing one reversal inside the draw noise). Pooled over
the three seeds, the drop from f = 0 to 1 is −0.45 ± 0.25, standard error 0.08, 11 of 11.

Paired against the published model with the fixed-scale map, pooled: +0.28 ± 0.30 (SE 0.09,
2 of 11) at f = 0, then +0.06 / +0.09 / +0.07 / +0.05 (SE 0.07-0.08, 4-7 of 11); per seed at
f = 1/4-3/4: seed 0 +0.08 to +0.10, seed 1 +0.10 to +0.16, seed 2 −0.01 to +0.03. Against the
mixture of specialists: +0.06 / −0.01 / −0.16 / −0.02 / −0.04 (SE 0.04-0.10, 3-7 of 11).
Against the published model as is: +0.22 at f = 0, −0.26 / −0.82 / −2.02 / −3.54 from f = 1/4
(8-11 of 11).

So with three seeds: the monotone decrease and the 0.4-0.5 drop from f = 0 to f = 1 are
robust (every seed, 11 of 11 environments pooled); the 0.1 gap to the fixed-scale map at
f ≥ 1/4 does **not** survive as a difference, it is +0.05 to +0.09 with standard errors of
0.08 and one seed at zero, i.e. V4 mixed equals the inference-time fix within the noise in
heterogeneous teams, except at f = 0 where it stays 0.28 behind (its 128 m weakness, 2 of 11);
the comparison with the mixture is a tie or a small advantage for V4 (−0.16 at f = 1/2).

Data `results_V45/results_hetero_V4mixed_s{1,2}.csv`; figure `fig_hetero_candidates.png`
now drawn with the three seeds.

**Follow-ups approved on 21 October (launched 22:30).** (i) Done above. (ii) V6 = V5 + V4
(`LPACMultiScaleGeometry`: V5's multi-scale maps into the CNN, V4's bounded-edge front layer
ahead of the unchanged TAGConv stack, the V4 class reused; 1,891,620 parameters = V5 +
84,258; unit test: with V5's weights and a zeroed layer output V6 computes exactly V5), a
3-epoch trial (1865590) then three seeds on mixed ranges in parallel (1865591, 1865594,
1865597; 25 epochs, best checkpoint, 11 and 40 environments), about 8-10 GPU-hours. The
(a)/(b) analysis will use the honest noise (the six V5-mixed runs) and the 40 new
environments as the primary set.

**Result of V6 (22 October, 10:30).** Three seeds, 25 epochs in 2 h 40 each (324 s per epoch);
best validation 0.465 / 0.460 / 0.462 at epochs 5 / 4 / 6 with the training loss at 0.36-0.42
there, then rising validation (0.52-0.54 at epoch 25) while the training loss falls to 0.19.
V6 overfits early, exactly like V5 (best at epochs 4-5) and unlike V4 (best at epochs 21-25,
validation 0.35): the map pathway sets the early validation minimum and the run is stopped
before the edge layer is trained as far as it is in V4.

| | 128 m | 256 m | 512 m | 768 m | 1024 m |
|---|---|---|---|---|---|
| V6 mixed, per seed, 40 new envs | 2.20 / 2.08 / 2.08 | 2.00 / 1.86 / 1.86 | 1.96 / 1.82 / 1.83 | 2.01 / 1.79 / 1.84 | 2.00 / 1.79 / 1.87 |
| V6 mixed, mean ± std, 40 new envs | 2.12 ± 0.07 | 1.91 ± 0.08 | 1.87 ± 0.08 | 1.88 ± 0.12 | 1.89 ± 0.11 |
| V6 mixed, mean ± std, 11 release envs | 2.12 ± 0.07 | 1.85 ± 0.05 | 1.83 ± 0.06 | 1.84 ± 0.10 | 1.85 ± 0.08 |
| V4 mixed (40 / 11) | 2.14 / 2.21 | 1.87 / 1.88 | 1.81 / 1.77 | 1.81 / 1.76 | 1.82 / 1.76 |
| V5 mixed, 6 runs (40 / 11) | 2.09 / 2.13 | 1.89 / 1.84 | 1.89 / 1.82 | 1.90 / 1.83 | 1.90 / 1.84 |

On the 40 new environments (primary): (a) V6 − V0 at 256 m +0.087 ± 0.228, SE 0.036, 15 of 40,
per-run gaps +0.18 / +0.04 / +0.04: no. (b) 1024 m − 256 m −0.021, SE 0.020, 24 of 40, per seed
0.00 / −0.07 / +0.01: flat, not significantly lower (V4 mixed −0.055, SE 0.016; V5 pool
+0.009). Paired, seed-averaged: V6 − V4 mixed −0.02 at 128 m (26 of 40), +0.04 at 256 m, +0.06 /
+0.06 / +0.07 at 512-1024 m (SE 0.02, 11-16 of 40); V6 − V5 mixed (six-run pool) +0.03 / +0.02 /
−0.02 / −0.02 / −0.01 (SE 0.01-0.02, 17-21 of 40); V6 − V0F +0.03 / 0.00 / −0.04 / −0.03 / −0.03;
V6 − centralized CVT −0.21 to −0.46 (32-38 of 40). On the 11 release environments: (a) +0.109,
SE 0.054, 2 of 11: no; (b) −0.002, SE 0.033: flat; V6 − V4 mixed −0.09 / −0.03 / +0.06 / +0.08 /
+0.09 (SE 0.04-0.06); V6 − V5 pool 0.00 to +0.02 (SE 0.02-0.04).

Reading. The gains do not add. V6 is V5: equal to the six-run V5 pool within 0.03 at every
range on both sets, flat beyond 256 m, 0.06-0.09 behind V4 mixed at 512-1024 m, and 0.09-0.11
above V0 at 256 m. With the multi-scale maps in the CNN, the model trains like V5 (validation
minimum at epoch 4-6, then overfitting), and the edge layer, which in V4 keeps improving the
fit until epoch 25, is cut off at a point where it has not yet contributed. By the agreed
order, (b) first, V4 mixed stays the best candidate; V6 is not evaluated in heterogeneous
teams. A V6 trained with early stopping on a different signal, or with the map pathway
regularized, might behave differently; that is a new recipe, not this one.

Data `results_V45/V6mixed_s{0,1,2}_{11,40}.csv`, loss histories `results_V45/train_V6mixed_s*/`;
figures `fig_v4v5.png` and `fig_v4v5_new40.png` (V6 added).

Figures `fig_v4v5.png` (11 release environments) and `fig_v4v5_new40.png`, all seeds; data
`results_V45/` (merged sweeps per seed and set, `*_v2_*` for the restarted V5 mixed runs,
loss histories of both launches, the first under `train_V5mixed_s*_partial/`).

### 14.x Checkpoint selection: epoch 25 instead of best validation (22-25 October)

Question: best-validation selection stops V5 and V6 mixed at epoch 4-6, before V6's bounded-edge
layer has trained as long as in V4 (best at 21-25). Does a late checkpoint let V6 improve with the
range? Evaluated `model_epoch025` for V5 mixed (full runs) and V6 mixed, and `model_epoch012` for V6,
same protocol (`lpac_study/checkpoint_epochs.py`). Cost / clairvoyant, mean of 3 seeds, 40 new envs
(primary) at 128 / 256 / 512 / 768 / 1024 m; b = paired 1024 - 256 m.

| Run | 128 | 256 | 512 | 768 | 1024 | b (SE, envs better) |
|---|---|---|---|---|---|---|
| V4 mixed (best val, ep. 21-25) | 2.14 | 1.87 | 1.81 | 1.81 | 1.82 | -0.055 (0.016, 27/40) |
| V5 mixed, best val (ep. 4-5) | 2.07 | 1.88 | 1.89 | 1.90 | 1.90 | +0.027 (0.023, 18/40) |
| V5 mixed, epoch 25 | 2.07 | 1.87 | 1.86 | 1.88 | 1.88 | +0.008 (0.027, 17/40) |
| V6 mixed, best val (ep. 4-6) | 2.12 | 1.91 | 1.87 | 1.88 | 1.89 | -0.021 (0.020, 24/40) |
| V6 mixed, epoch 12 | 2.10 | 1.89 | 1.90 | 1.89 | 1.89 | +0.004 (0.020, 23/40) |
| V6 mixed, epoch 25 | 2.10 | 1.87 | 1.85 | 1.87 | 1.90 | +0.023 (0.023, 19/40) |

Paired on 40 envs: V6@25 - V6@best within +-0.04 (SE 0.02-0.03) at every range; V5@25 - V5@best
within +-0.03; V6@25 - V4 mixed -0.04 / 0.00 / +0.05 / +0.06 / +0.08 (V4 better beyond 512 m);
V6@25 - V5@25 within +-0.03. On the 11 released envs V6@25 reads 2.10 / 1.79 / 1.75 / 1.75 / 1.80
(b = +0.011, SE 0.044, 4/11), again flat.

Answer: no. The checkpoint does not matter: from epoch 5 to 25 the validation MSE rises from 0.46
to 0.54 while the coverage cost does not move, so validation MSE carries no information about the
cost for these runs. V6 does not improve with the range at any checkpoint, and the V4 and V5 gains
do not add up: V6 = V5. V4 mixed remains the only trained model whose cost falls with the range.
Checkpoint rule used elsewhere: best validation loss for every run (V0, V0F, V4, V5, V6), as in the
authors' `train_launch`.

## 15. Running and next (as of 14 October)

Nothing queued on the GPU partition as of 17 October 21:30; all chains of section 11 retrieved.
Nothing running or queued on FarmShare as of 22 October 11:00; every chain of section 14 is retrieved.

Next, in order (1 and 2 done, section 9; 4 done, section 10):

3. Information loss vs distribution shift in the map: same neighbors in a finer map.
4. Done (section 10): the map-less variants lose 0.75 at 256 m and the edge-geometry
   GATv2 does not recover it. Possible follow-ups, not started: tune V2 (learning rate,
   normalization, residual connections) or give it a range-independent graph (edges capped
   at 256 m at inference, the graph analogue of the fixed-scale map).
5. Optionally, LPAC trained at 512 m to connect with Fig. 20.

Contact: an email with the results and three questions (other checkpoints, whether the
neighbor-map sensitivity is known, scope and compute) was sent to Charilaos Kanatsoulis,
co-instructor and co-author of Muthusamy et al., in the week of 12 October.
