# Communication range and the LPAC swarm policy

CS 224W (Machine Learning with Graphs), Stanford, Autumn 2026 — course project.

LPAC (Agarwal et al., IEEE T-RO 2025) is a decentralized coverage-control policy for robot
swarms: each robot runs a CNN on its local maps, a GNN (5 × `TAGConv`, K = 3) over the
communication graph, and an MLP that outputs a velocity. This project studies what happens
when the policy is run with a larger communication range than it was trained with.

**Status (14 October 2026)**

- The published LPAC-K3 checkpoint, trained at 128 m, is the best controller at 128 and
  256 m, then its cost triples by 1024 m and it becomes worse than the non-learned CVT
  baselines, in all 11 released environments.
- A model we retrained ourselves with the authors' recipe (100,000 samples, 2 h 35 on one
  FarmShare L40S GPU) shows the same collapse; two training seeds agree.
- At inference, capping the graph given to the GNN does not help. Drawing the neighbor map
  given to the CNN at a fixed 256 m scale restores the 256 m performance at every range:
  the failure comes from the rasterized neighbor maps, not from the graph.
- A model trained at 256 m degrades much less (2.31 at 1024 m instead of 4.4 to 5.2) but
  still degrades: the training range sets the neighborhood regime the map can read.
- Three training seeds agree to ±0.02 at 256 m and ±0.26 at 1024 m. Sampling the training
  data every 20 steps instead of 5 flattens the curve (3.0 beyond 768 m) at the price of a
  worse optimum (2.01 at 256 m); the authors' recipe is kept for all comparisons.
- Validated on 40 new environments and on all four retrained models: the map drawn at a
  256 m scale is the best level for every model (1.7-1.9 from 256 m to 1024 m, beating the
  centralized CVT in 36-39 of 40 environments), and 256 m is the optimum of a scale sweep
  on both environment sets. Below 256 m the fixed-scale map can only contain the robots in
  radio range, and it performs as the as-is map does (within noise).
- Variants trained with the same recipe (3 seeds each): V1 without neighbor maps (2-channel
  CNN) loses 0.75 at 256 m (2.48 vs 1.74) and V2 = V1 with the relative positions of
  neighbors as edge attributes of a 5-layer GATv2 GNN is worse still (2.96 at 256 m,
  underfits). Both are flat across ranges (3.1-3.3 at 1024 m) but never beat the centralized
  CVT beyond 128 m: the rasterized map carries information the graph alone does not
  recover with this recipe. Same picture on the 40 new environments.
- V0 trained on 100k samples mixing 128/256/512/1024 m (3 seeds): no collapse (1.9-2.0 at
  512-1024 m, beats the centralized CVT from 256 m) but 2.08 at 256 m instead of 1.74: it trades
  the 256 m optimum for robustness, which the fixed-scale map gets for free.
- V3 = V1 with a PointNetConv layer reading relative positions at a fixed 256 m scale (3 seeds,
  100 epochs): the best map-less model, 2.0 at 256 m (V0 1.74, V1 2.48) and only 2.6 at 1024 m
  (V0 4.6); still 0.5-1.1 behind the fixed-scale map beyond 512 m. An 11-epoch CPU run of it had
  collapsed (5.8 at 1024 m): under-training, not architecture.
- Heterogeneous teams at inference (a fraction f of robots hearing to 1024 m, the others 128 m):
  the cost grows convexly with f (2.0, 2.4, 2.9, 4.0, 5.3 over 8 draws), a few far-hearing robots do
  not drag the team down; the graph changes the other robots' actions by 20-36 % but the damage comes
  from the far-hearing robots' own map; with the map fixed at 256 m the cost is flat (1.7-2.1) at
  every f. Which robots hear far changes the cost by 0.3-0.5 across draws, unpredictably from where
  they start (the collapsed regime is sensitive to any change of the initial condition).
- Fig. 20 of the paper, one model per training range evaluated at its own range (seed 0): 2.15 /
  1.76 / 1.79 / 1.80 at 128 / 256 / 512 / 1024 m, below both CVTs everywhere (centralized 2.52): the
  collapse is a property of the 128 m checkpoint out of its range, not of the architecture at long
  range. The paper's rise up to 512 m and moderate decline after are not seen: flat from 256 m.
- Mixture of range specialists at inference (each robot of a 128/1024 m team runs the model trained
  at its own range, layer by layer on the mixed inputs): at the level of the better specialist used
  alone (−0.05 to +0.23 with specialists from independent seeds; a common training seed helps by
  0.1-0.3), never much worse, so no sign of incompatible message languages; still 0.1-0.4 behind
  the fixed-scale map on the published model.
- V0 trained with the fixed-scale map on mixed ranges (3 seeds): flat, 1.86 ± 0.01 at 256 m and
  1.81-1.83 from 512 m, equal to the inference-time fix from 512 m; 0.12 ± 0.05 (standard error)
  above V0 at 256 m with three seeds on each side, so the "within 0.05" criterion is not met and
  the gap belongs to the recipe. The same recipe on the 128 m data alone fails (2.0-2.1
  everywhere): the training data must cover the map's content.
- V3 trained on mixed ranges (seed 0) fails: 2.6-2.9 at every range, above the centralized CVT and
  0.5 above V3 trained at 128 m at short range; one recipe for inputs four times apart fits neither.
- V4 (V3 with bounded edge features [(p_j − p_i)/d, tanh(d/256 m)]) trained on mixed ranges (3 seeds):
  1.88 at 256 m, 1.76 at 1024 m, the first model whose cost keeps falling beyond 256 m (−0.12 paired,
  10 of 11 environments), equal to the fixed-map recipe and below the centralized CVT everywhere;
  trained at 128 m alone (3 seeds) it does not collapse (1.9-2.0 at every range) and beats V3 at
  every range. Still 0.14 above V0 at 256 m.
- V6 = V5's maps + V4's edge layer (3 seeds, mixed ranges) does not add the two gains: it trains and
  behaves like V5 (early overfitting, flat 1.87-1.91 beyond 256 m, equal to V5 within 0.03, 0.06-0.09
  behind V4 mixed at long range); V4 mixed remains the only model that improves with the range.
- In heterogeneous 128/1024 m teams, V4 mixed (3 seeds) is the first model whose cost falls
  monotonically with the fraction of far-hearing robots (2.21 to 1.76, −0.45 ± 0.08, every seed),
  beating V0R everywhere and matching both the mixture of specialists and the fixed-scale map on the
  published model within the noise (except at f = 0, where it is 0.28 behind).
- V5 (three fixed-scale map pairs, 256/512/1024 m) trained on mixed ranges: 1.84 ± 0.08 at 256 m over
  six runs (two launches of three seeds, which differ by up to 0.15 at a fixed seed: GPU training is
  not deterministic), 0.10 above V0, flat 1.82-1.84 up to 1024 m, equal to the single fixed-scale map
  within the run noise; it does not improve with range. Trained at 128 m alone (one run) it reaches
  1.77 at 256 m with no collapse (1.97 at 1024 m). The "within 0.05 of V0" margin is below the
  run-to-run noise of this protocol (about 0.08 at 256 m on the 11 release environments).
- One model with the robot's own range as a GNN input, trained on teams with a range drawn per robot
  (V0R, seed 0): no collapse (1.9-2.1 from 512 m) but equal to plain mixed-range training within
  0.1 and 0.4-0.6 worse than the 128 m model at 128-256 m; in heterogeneous teams it beats the
  published model from f = 1/2 but stays 0.2-0.4 above the published model with the fixed-scale map.

- Keeping the far robots in the fixed-scale map, projected onto the 256 m circle, is worse than
  dropping them: on the 11 release environments the published model reads 2.03 / 2.55 / 2.64 at
  512 / 768 / 1024 m against 1.78 / 1.76 / 1.71 when they are dropped (paired +0.25 to +0.94, better
  in at most 2 of 11 environments), though both beat the published map (2.77 / 4.52 / 5.22). The
  projection piles up to 30 robots onto the border ring, with cell values up to 4.29 against 1.10,
  so what breaks the policy at long range is the distortion of the map, not the loss of the far
  robots.

Details, numbers and caveats: [`docs/journal.md`](docs/journal.md). Framing document with
figures (French): [`docs/framing.html`](docs/framing.html), also online at
<https://claude.ai/artifact/LcHe2tRjWD1ZMXqiXVh6zV>.

## Repository layout

| Path | Content |
|---|---|
| `docs/journal.md` | project log: decisions, measurements, corrections, next steps |
| `docs/framing.html` | framing document (French) with diagrams |
| `docs/conversations/` | raw exports of the working conversations |
| `RAPPORT*.md` | one-page run reports, in French: range sweep (`RAPPORT.md`), graph × map factorial (`RAPPORT_2x2.md`), fixed-scale map, empty map and sensitivity (`RAPPORT_M.md`), training trial (`RAPPORT_TRAIN.md`) |
| `results.csv`, `results/` | range sweep of the published model: one row per (environment, range, team size, controller); effective rank per layer in `results/rank.csv` |
| `results_2x2/`, `results_2x2.csv` | inference-time factorial (graph levels G0–G1 × neighbor-map levels C0–C4), noisy repeats |
| `results_train/`, `training/` | training trial: configs, Slurm scripts, losses, timings, evaluations of retrained models |
| `results_variants/` | variants V1/V2: merged evaluations per seed on the 11 and 40 environments, loss histories, configs |
| `results_night2/` | capped-graph confirmation for V1/V2, mixed-range model evaluations, loss histories and dataset parameters |
| `results_hetero/`, `farmshare_hetero.sbatch` | heterogeneous ranges at inference (`lpac_study.heterogeneous`): per-robot ranges, frozen-state probe |
| `results_fig20/`, `fig_fig20*.png` | one model per training range (128/256/512/1024 m) evaluated at every range: the paper's Fig. 20 |
| `results_h1/`, `farmshare_mixture.sbatch`, `fig_mixture.png` | mixture of range specialists in a heterogeneous team (`lpac_study.mixture`) |
| `results_A/`, `fig_fixedmap_training.png` | V0 trained with the fixed-scale neighbour map (mixed ranges and 128 m control) |
| `results_B/` | V3 trained on the mixed-range dataset |
| `results_clamp/` | question Q1 of the teaching assistant: the fixed-scale map that keeps the far robots on the border circle (`K256`) against the one that drops them (`C3`) |
| `results_H2/`, `fig_range_input.png` | V0R: the robot's own range as GNN input, trained on per-robot ranges; homogeneous sweeps and heterogeneous grid |
| `fig_*.png` | figures; `fig_email.png` is the self-contained summary figure; `fig_variants*.png` the variant comparison |
| `lpac_study/` | Python package (PyTorch Geometric) used by all scripts and the notebook |
| `lpac_study/variants.py` | model variants V1 (no neighbor map), V2 (edge geometry, GATv2), V3 (V1 + PointNetConv front layer), V0R (V0 + own range as GNN input), V4 (V3 with bounded edge features) and V5 (multi-scale fixed maps) as subclasses of the library model; `--variant` on `training/train_launch.py`, `lpac_study.sweep`, `lpac_study.factorial` and the Slurm scripts |
| `lpac_study/heterogeneous.py`, `lpac_study/mixture.py` | heterogeneous teams at inference (range per robot, frozen-state probe, start-position descriptors) and the mixture of range specialists |
| `training/redraw_ranges.py` | rebuilds the range-dependent tensors of a dataset at another range, at a fixed map scale, at three fixed scales (V5), or with a range per robot |
| `tests/` | unit tests of the V4 edge features and the V5 maps |
| `results_V45/`, `fig_v4v5*.png`, `fig_hetero_candidates.png` | V4 and V5 evaluations per seed on both environment sets, loss histories, V4 mixed in heterogeneous teams |
| `lpac_eval.ipynb` | self-contained Colab notebook (installs, downloads, sweeps, plots) |
| `run_all.sh` | reruns the range sweep from scratch on Linux/WSL |
| `farmshare_factorial.sbatch` | Slurm job for the factorial on FarmShare (Intel nodes only) |

### Condition codes used in the results

| Code | Meaning |
|---|---|
| G0 | graph given to the GNN: all robots within communication range (as published) |
| G1 | graph capped to each robot's k nearest neighbors within range (undirected) |
| R\<m\> | graph with edges only between robots within min(m, test range) metres (R256 = capped at 256 m) |
| C0 | neighbor map given to the CNN: as produced by the library (as published) |
| C1 | map drawing only the k nearest neighbors, at the test-range scale |
| C2 | map drawn as at training: neighbors within 128 m, 128 m scale |
| C3 | map drawn at a fixed 256 m scale: neighbors within 256 m |
| S\<m\> | map drawn at a fixed \<m\> metre scale: neighbors within min(m, test range) (S256 = C3) |
| K\<m\> | same fixed scale, but the robots beyond \<m\> metres are projected onto the \<m\> metre circle and written into the border cell instead of being dropped |
| C4 | empty map |

## Reproducing

Everything except training runs on CPU under Linux or WSL. Data, model checkpoints and the
release archive are kept outside the repository (`~/lpac` by default).

```bash
bash run_all.sh                     # venv, release archive, range sweep, merge, figures
```

Step by step, with `LPAC_DATA_DIR` pointing to the extracted `lpac/` folder of the
CoverageControl v1.3.1 release:

```bash
python -m lpac_study.sweep --out-dir results/raw
python -m lpac_study.merge --parts results/raw --out-dir results
python -m lpac_study.plots --results-dir results --fig-dir .

python -m lpac_study.checks --comm-range 1024
python -m lpac_study.factorial --baseline results/results.csv --out-dir results_2x2 \
    --ranges 1024 --cells G0C0 G0C1 G0C2 G1C0 G1C1 G1C2 --k 4
python -m lpac_study.plots_factorial --results results_2x2/results_2x2.csv \
    --baseline results/results.csv --fig-dir .
```

Training uses the authors' scripts (`python/data_generation`, `python/training` at tag
v1.3.1) through `training/train_launch.py` and the Slurm files in `training/`; see
`RAPPORT_TRAIN.md`.

Reproducibility notes: evaluations are bit-identical across Intel CPUs but differ by up to
5.6 % per environment on AMD CPUs; the simulator draws new environments from
`std::random_device`, so generated environments must be saved to be reused.

## Package `lpac_study`

| Module | Content |
|---|---|
| `paths.py` | locates the release files, no hard-coded paths |
| `simulation.py` | parameters (range, team size), environments, normalized cost |
| `lpac_policy.py` | the published model, hooks on the 5 `TAGConv` layers, access to the PyG `Data` |
| `graph_stats.py` | graph degree via `torch_geometric.utils.degree` |
| `effective_rank.py` | effective rank (exponential of the entropy of normalized singular values) |
| `episode.py` | one episode of one controller, with measurements |
| `sweep.py` | range and constant-degree sweeps, resumable CSV |
| `merge.py`, `plots.py` | merging and figures |
| `neighbourhood.py` | k-NN graph bounded by range, reconstructed neighbor maps |
| `checks.py` | checks that reconstructed inputs match the library |
| `factorial.py` | graph × map factorial at inference (levels C0–C4, noisy repeats) |
| `plots_factorial.py` | `fig_2x2.png`, `fig_dose.png`, `fig_fix.png` |

## References

- Agarwal, Muthukrishnan, Gosrich, Kumar, Ribeiro. *LPAC: Learnable Perception-Action-Communication Loops with Applications to Coverage Control.* IEEE T-RO, 2025. [arXiv:2401.04855](https://arxiv.org/abs/2401.04855)
- Code and checkpoint: [KumarRobotics/CoverageControl](https://github.com/KumarRobotics/CoverageControl), release v1.3.1.
- Muthusamy, Owerko, Kanatsoulis, Agarwal, Ribeiro. *Generalizability of Graph Neural Networks for Decentralized Unlabeled Motion Planning.* 2024. [arXiv:2409.19829](https://arxiv.org/abs/2409.19829)
- Owerko, Vatnsdal, Agarwal, Kumar, Ribeiro. *MAST: Multi-Agent Spatial Transformer for Learning to Collaborate.* 2025. [arXiv:2509.17195](https://arxiv.org/abs/2509.17195)
