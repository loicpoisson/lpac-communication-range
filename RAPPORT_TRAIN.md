# Rapport — essai d'entraînement chronométré de LPAC (12 et 13 octobre 2026)

**Conclusion : entraînement complet réaliste**, sur FarmShare (gratuit) avec un GPU L40S. À l'échelle du papier
(100 000 points, 100 époques) : génération 25 min en parallèle, entraînement **2 h 35**, évaluation 4 min. Le modèle
obtenu est équivalent au modèle publié à 256 m (écart apparié +0,05 ± 0,16) et bat le CVT centralisé dans 11
environnements sur 11. Sans GPU, seule une échelle réduite tient en moins de 6 h ; le portable ne permet rien de sérieux.

Architecture, scripts et hyperparamètres inchangés (CoverageControl v1.3.1 : `data_generation.py`, `CNNGNNDataset`,
`TrainModel`, `learning_params.toml` de la release : LR 1e-4, weight decay 1e-3, batch 100, cartes de voisinage comprises).
Seul ajout : `training/train_launch.py`, qui reprend `train_lpac.py` avec graines fixées (torch, numpy, random ; graine 0),
un checkpoint par époque et le chronométrage, en appelant `TrainModel.train_epoch` / `validate_epoch` tels quels.
*Limite : le simulateur tire ses environnements de `std::random_device`, la génération n'est pas reproductible par graine ;
les fichiers générés font foi et restent sur FarmShare.*

## Temps mesurés

| Mesure | Portable WSL (i7-1065G7, 2 threads torch) | FarmShare Intel (Xeon 6148) | FarmShare GPU (L40S) |
|---|---|---|---|
| Génération, par échantillon (1 processus) | 1,4 s (200 éch. en 285 s) | 0,55 à 0,73 s (2 000 éch. en 1 090 à 1 460 s, 50 tâches) | — |
| RAM du générateur | 2,9 Go (200 éch.) | 4,8 Go (2 000 éch.) : ~0,5 Mo/éch. + 1,6 Go fixes | — |
| Époque, 140 échantillons | 45 à 65 s | 4 à 7 s (8 cœurs) | 0,4 s (6 s la première) |
| Époque, 28 000 train + 8 000 val | non mesuré | 676 s + 86 s (16 cœurs) | 22,4 s + 6,1 s |
| Époque, 70 000 train + 20 000 val | non mesuré | 1 935 à 2 304 s + ~205 s (32 cœurs) | 74 à 84 s + 11 à 12 s |
| Chargement du jeu en RAM (40k / 100k) | — | 85 s / 202 s | 78 s / 212 s |
| Attente d'un GPU (file partagée) | — | — | 10 min, 25 min la nuit ; 3 h 10 en journée |

Un « dataset » du script = **un échantillon** = l'état des 32 robots à un pas de temps (cartes locales 32 × 32, cartes de
voisinage, positions, actions du clairvoyant), prélevé tous les 5 pas d'un épisode (`EveryNumSteps = 5`), plus 10 %
d'états convergés. Les « 100 000 data points » du papier = `NumDataset = 100000`, soit ~250 épisodes distincts. Disque :
0,37 Mo par échantillon (tenseurs creux) ; 15 Go pour 40 000 et ~37 Go pour 100 000, parties et jeu fusionné compris.

## Ce qui a tourné, et où

| Étape | Résultat | Où |
|---|---|---|
| 1. Génération 200 éch., chargement `CNNGNNDataset` | OK (140/40/20), 75 Mo | WSL |
| 2. Entraînement 3 époques | portable, CPU FarmShare, GPU FarmShare (tableau) | WSL + FarmShare |
| 3a. 40 000 éch. (20 tâches × 2 000), fusion (`merge_datasets.py`, 2 min 43, 28k/8k/4k) | OK | FarmShare, nœuds Intel |
| 3b. Entraînement 40k, GPU, 100 époques | 49 min ; meilleure val. **0,523 à l'époque 5**, puis sur-apprentissage (ép. 100 : train 0,148, val 0,639) | oat-05 |
| 3c. Entraînement 40k, CPU 16 cœurs, 20 époques | 4 h 14 ; meilleure val. 0,522 à l'époque 5, même trajectoire | wheat-01 |
| 3d. 100 000 éch. (50 tâches), fusion (70k/20k/10k) | OK | FarmShare, nœuds Intel |
| 3e. Entraînement 100k, GPU, 100 époques | **2 h 35** ; meilleure val. **0,449 à l'époque 13** (papier : époque 16) ; ép. 100 : train 0,203, val 0,530 | oat-05 |
| 3f. Entraînement 100k, CPU 32 cœurs, 12 époques | 7 h 30 (2 035 s + 216 s par époque) ; meilleure val. 0,451 (époque 12, plateau depuis l'époque 9) | wheat-03 |
| 4. Évaluations, 11 environnements, 128 et 256 m | tableau ci-dessous ; 4 min par modèle (11 tâches) | FarmShare, nœuds Intel |
| Suites lancées (13 oct., 18 h 45) | balayage 128 à 1024 m du modèle 100k (fait, ci-dessous) ; deuxième graine du run 100k et modèle entraîné à 256 m (faits, ci-dessous) ; troisième graine et jeu décorrélé (`EveryNumSteps = 20`) faits dans la nuit du 13 au 14 (voir `docs/journal.md`, section 8 : trois graines à ±0,02 à 256 m et ±0,26 à 1024 m ; le jeu décorrélé donne 2,01 à 256 m et ~3,0 au-delà de 768 m) | FarmShare, chaînés par Slurm |

Courbes : `fig_train_loss.png`. Avec 28 000 points d'entraînement, le minimum de validation arrive à l'époque 5 et le
modèle mémorise ensuite ; avec 70 000, le minimum est à l'époque 13 et vaut 0,449, dans la plage du papier (0,35 à 0,45
lu sur sa figure 6). Le checkpoint retenu est celui de meilleure validation, comme chez les auteurs.

## Évaluation (coût moyen / clairvoyant, 11 environnements, 600 pas, protocole de `results.csv`)

| Modèle | 128 m | 256 m | différence appariée au modèle publié (128 / 256 m) |
|---|---|---|---|
| publié (`model_k3_1024_state_dict.pt`) | 1,98 ± 0,46 | 1,69 ± 0,18 | — |
| **100k, GPU, époque 13 (meilleure val.)** | 2,15 ± 0,51 | **1,74 ± 0,25** | +0,17 ± 0,38 (7/11 moins bon) / **+0,05 ± 0,16 (6/11)** |
| 100k, CPU 32 cœurs, époque 12 (meilleure val.) | 2,17 ± 0,50 | 1,76 ± 0,27 | +0,19 ± 0,35 (8/11) / +0,07 ± 0,17 (6/11) |
| 100k, CPU, époque 5 (intermédiaire) | 2,15 ± 0,47 | 1,77 ± 0,28 | +0,16 ± 0,34 (6/11) / +0,08 ± 0,17 (7/11) |
| 40k, GPU, époque 5 (meilleure val.) | 2,10 ± 0,32 | 1,88 ± 0,24 | +0,12 ± 0,31 (7/11) / +0,19 ± 0,11 (10/11) |
| 40k, GPU, époque 100 (sur-appris) | 2,40 ± 0,59 | 1,85 ± 0,20 | +0,42 ± 0,65 (9/11) / +0,17 ± 0,13 (10/11) |
| 40k, CPU 16 cœurs, époque 5 | 2,06 ± 0,34 | 1,87 ± 0,28 | +0,07 ± 0,36 (7/11) / +0,18 ± 0,17 (10/11) |
| CVT centralisé / décentralisé | 2,52 / 2,59 | 2,52 / 2,57 | |

Le modèle 100k bat le CVT centralisé dans **11 environnements sur 11** aux deux portées ; à 256 m il est indiscernable
du modèle publié (différence appariée +0,05, écart-type 0,16, moins bon dans 6 env. sur 11) ; à 128 m il reste un peu
derrière (+0,17, dans le bruit d'un épisode, ~0,35 pour ce modèle). L'évaluation chaînée sur `model_best` redonne
exactement celle de l'époque 13 (écart 0,0). Le passage de 40k à 100k vaut −0,14 à 256 m. Nous savons donc refaire un LPAC
équivalent au publié ; l'écart résiduel à 128 m n'est pas interprété ici (une graine, un tirage de données).

Checkpoints (hors OneDrive) : WSL `~/lpac/train/farmshare_models/{models100k_gpu,models40k_gpu,models_cpu20}/` (meilleur,
époque 13 ou 5, dernier) ; FarmShare `/home/users/lpoisson/lpac/train/large/models{100k,,_cpu,100k_cpu}/` (toutes les époques) ;
données `/home/users/lpoisson/lpac/train/large/{data,data100k}`.

## Le modèle réentraîné sous le balayage de portée (`fig_range_retrained.png`)

| Portée | 128 m | 256 m | 512 m | 768 m | 1024 m |
|---|---|---|---|---|---|
| **100k, graine 0, époque 13** | 2,15 ± 0,51 | **1,74 ± 0,25** | 2,49 ± 0,32 | 3,46 ± 0,71 | 4,36 ± 1,08 |
| 100k, graine 1, époque 9 (40 époques, 1 h 03) | 2,14 ± 0,46 | 1,73 ± 0,21 | 2,51 ± 0,30 | 3,54 ± 0,64 | 4,58 ± 1,08 |
| apparié (graine 1 − graine 0) | −0,01 ± 0,29 | −0,01 ± 0,17 | +0,02 ± 0,16 | +0,08 ± 0,40 | +0,22 ± 0,89 |
| publié | 1,98 ± 0,46 | 1,69 ± 0,18 | 2,77 ± 0,38 | 4,52 ± 1,14 | 5,22 ± 1,41 |
| apparié (réentraîné − publié) | +0,17 ± 0,38 | +0,05 ± 0,15 | −0,27 ± 0,34 (2/11 moins bon) | −1,07 ± 0,58 (0/11) | −0,85 ± 1,01 (1/11) |
| < CVT centralisé (2,52) | 11/11 | 11/11 | 4/11 | 3/11 | 0/11 |

Le modèle réentraîné sur nos propres données reproduit le phénomène de bout en bout : optimum à 256 m, puis dégradation
monotone jusqu'à 4,36 à 1024 m, sous les deux CVT dès 512 m. La dégradation est un peu moins forte que celle du
checkpoint publié (−0,85 à 1024 m, −1,07 à 768 m, cohérent dans 10 à 11 environnements sur 11), ce qui dit que
l'ampleur exacte dépend du tirage d'entraînement, mais pas la forme. Degrés mesurés identiques au balayage initial
(1,4 / 4,5 / 12,4 / 19,6 / 25,1).

**Variabilité entre graines.** Deux entraînements identiques (mêmes 100 000 points, graines torch 0 et 1 ; meilleure
validation 0,449 à l'époque 13 et 0,450 à l'époque 9) donnent des balayages indiscernables : différences appariées de
−0,01 à +0,22, dans le bruit d'un épisode. L'écart avec le checkpoint publié au-delà de 512 m (−0,85 à −1,07) dépasse
donc la variabilité de notre entraînement ; il vient du tirage de données ou du protocole d'entraînement des auteurs
(deux GPU), pas d'une graine. Pour la phase 2, une différence entre variantes inférieure à ~0,1 en moyenne sur les 11
environnements n'est pas interprétable sans répétition.

## Un modèle entraîné à 256 m (`fig_range_retrained.png`)

Même recette, mais les 100 000 points sont générés avec `CommunicationRange = 256` (cartes de voisinage et graphe
d'entraînement à 256 m, 4 à 5 voisins par robot). Entraînement GPU arrêté par la limite de 3 h à l'époque 35 (265 s par
époque : le chargeur, pas le GPU) ; meilleure validation **0,391 à l'époque 7**, nettement plus basse qu'à 128 m (0,449),
l'imitation étant plus facile avec plus de voisins.

| Portée | 128 m | 256 m | 512 m | 768 m | 1024 m |
|---|---|---|---|---|---|
| entraîné à 256 m | 2,28 ± 0,50 | 1,76 ± 0,28 | **1,74 ± 0,17** | 1,90 ± 0,24 | 2,31 ± 0,24 |
| entraîné à 128 m (graine 0) | 2,15 ± 0,51 | 1,74 ± 0,25 | 2,49 ± 0,32 | 3,46 ± 0,71 | 4,36 ± 1,08 |
| apparié (256 m − 128 m) | +0,13 ± 0,23 | +0,02 ± 0,23 | −0,75 ± 0,25 (11/11) | −1,56 ± 0,71 (11/11) | −2,06 ± 1,03 (11/11) |
| < CVT centralisé (2,52) | 8/11 | 10/11 | 10/11 | 9/11 | 6/11 |

Le modèle entraîné à 256 m est aussi bon que le modèle 128 m à sa propre portée (1,76 contre 1,74), un peu moins bon
à 128 m (il a moins de voisins qu'à l'entraînement), et **bien plus robuste au-delà** : 2,31 à 1024 m au lieu de 4,4 à 5,2,
mieux que le modèle 128 m dans 11 environnements sur 11 dès 512 m. Il reste toutefois une dégradation (1,74 → 2,31 de
512 à 1024 m), et à 1024 m il ne bat le CVT centralisé que dans 6 environnements sur 11. Lecture : la portée
d'entraînement fixe le régime de voisinage que la carte sait lire ; en élargir le régime recule la chute sans la supprimer.
C'est le repère pour la phase 2 : une variante sans carte devra faire au moins 2,3 à 1024 m, et idéalement rester plate.

## Extrapolation et ce qui est réaliste

| | Génération 100k | Entraînement 100 époques | Total |
|---|---|---|---|
| Portable WSL | ~39 h, impossible en un processus (RAM ~50 Go) | ~680 h (0,35 s/éch./époque) | non réaliste |
| FarmShare CPU, 32 cœurs Intel | 15 à 20 heures-cœur, ~25 min sur 50 tâches | ~62 h (37 min/époque, mesuré) | non réaliste en < 6 h ; 12 époques = 7 h 30 |
| FarmShare GPU L40S | idem | **2 h 35 mesuré** (110 Go de RAM, 12 cœurs) | **~3 h** + attente du GPU |

L'extrapolation linéaire en échantillons s'est vérifiée (40k → 100k : 22 s → 80 s par époque GPU, ratio 3,6 pour 2,5×
les données, le chargeur dominant). L'aléa principal est la file GPU : 10 à 25 min la nuit, plus de 3 h en journée ;
soumettre le soir, avec `--dependency`, et demander peu (12 cœurs, 110 Go, 3 h) pour passer en backfill. En moins de 6 h :
avec GPU, l'échelle du papier ; sans GPU, 100 000 points × 10 époques (~6 h 15 sur 32 cœurs) ou 40 000 × 20 (4 h 15), ce qui
suffit puisque le minimum de validation arrive avant l'époque 15.

## Fichiers

`training/train_launch.py`, `merge_datasets.py`, `plot_loss.py`, `farmshare_generate.sbatch`, `farmshare_train.sbatch`,
`farmshare_eval.sbatch` ; configurations dans `training/configs/` ; chronométrages et pertes dans `training/logs/` ;
évaluations par environnement dans `results_train/` (`results_eval*.csv`).
