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
128 m (jobs 1847989-1847990); the 3-seed, 40-environment numbers are those of section 10
at 256 m (V1 2.48 / 2.54, V2 2.96 / 2.97). Answer to the question asked: capping the graph
puts V2 back exactly on its 256 m value, so the whole degradation of V2 from 256 to
1024 m (2.96 to 3.31) comes from the extra edges, whose attributes lie outside the training
range; for V1 (2.48 to 3.14) it comes from the graph density alone. Capping does not make
either variant good: 2.96 and 2.48 remain above the centralized CVT (2.52) or at it.

**3. V3 = V1 + one `PointNetConv` layer in front of the unchanged TAGConv stack**
(`LPACPointNet` in `lpac_study/variants.py`, every choice commented in the class
docstring). The layer reads ``[x_j, (p_j - p_i) / 256 m]`` on every edge, max-aggregates,
and its output is added to the TAGConv input (residual); 1,889,636 parameters (V1 +
84,002), receptive field 16 hops. Seed 0 only, 100k samples at 128 m, queued after the
mixed seed-0 training (job 1848021) with its evaluations (1848022-1848023). Control before
reading the evaluation: the training loss must reach V1's level (about 0.2 at epoch 100);
if not, seeds 1 and 2 stay unlaunched and it is reported as such.

Generation and training times are read from the Slurm accounting and the job logs.

## 12. Running and next (as of 14 October)

Running on FarmShare (submitted 15 October 00:40): the three chains of section 11.

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
