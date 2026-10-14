# Rapport — mesures M1, M2, M3 (carte à l'échelle 256 m, carte vide, sensibilité)

Même protocole que le factoriel : modèle publié, 32 robots, 11 environnements, 600 pas, torch mono-thread, mêmes
positions initiales ; seules les entrées du modèle changent ; coût moyen sur l'épisode divisé par le clairvoyant.
Lignes ajoutées à `results_2x2.csv` (638 lignes ; colonne `repeat` : 0 = épisode nominal, 1 à 5 = répétitions
bruitées de M3, avec `noise_sigma` et `clairvoyant_cost_mean` recalculé sur le départ perturbé).

**Où ça a tourné.** Tout sur FarmShare (Slurm, une tâche par environnement, `farmshare_factorial.sbatch`), sauf les
contrôles préalables sur WSL. Les tâches tombées sur les nœuds AMD (rye, barley) ont été **rejetées et recalculées sur les
nœuds Intel** : avec les mêmes entrées, les noyaux de convolution y arrondissent autrement et la trajectoire diverge
(jusqu'à 5,6 % sur le coût moyen d'un environnement, 1e-9 sur d'autres), alors que les nœuds Intel AVX-512 reproduisent
le portable au bit près (contrainte `CPU_MNF:INTEL` ajoutée au script). Ce fait est en lui-même une mesure de
sensibilité, cohérente avec M3.

**Contrôle M1 (obligatoire).** G0+C3 à 256 m redonne le modèle tel quel sur les 11 environnements : écart maximal
du coût moyen **0,0** (après avoir calculé les positions relatives en double comme le C++ ; en float32 l'écart était de
4e-9). La carte sans voisin de la bibliothèque est bien nulle (vérifié sur 14 robots isolés par pas à 128 m).

## M1 — carte à l'échelle de 256 m (C3), 11 environnements, moyenne ± écart-type

| Portée | modèle tel quel | G0+C3 | G1+C3 |
|---|---|---|---|
| 256 m | 1,69 ± 0,18 | 1,69 ± 0,18 (contrôle, identique) | — |
| 512 m | 2,77 ± 0,38 | 1,78 ± 0,20 | 1,75 ± 0,26 |
| 768 m | 4,52 ± 1,14 | 1,76 ± 0,16 | 1,71 ± 0,21 |
| 1024 m | 5,22 ± 1,41 | **1,71 ± 0,19** | **1,71 ± 0,21** |

Comparaison appariée par environnement, G0+C3 à 1024 m moins le modèle tel quel à 256 m : différences
−0,05, +0,07, +0,05, −0,23, +0,09, +0,07, +0,10, +0,33, +0,12, −0,18, −0,17 ; **moyenne +0,02, écart-type 0,16,
erreur standard 0,05**, positive dans 7 environnements sur 11. Pour G1+C3 : +0,02 ± 0,20, 8/11. Pour mémoire, G1+C2
(échelle 128 m) : +0,49 ± 0,52, 10/11 ; modèle tel quel à 1024 m : +3,53 ± 1,39, 11/11.
*Conclusion M1 : dessiner la carte de voisinage à l'échelle de 256 m rend au modèle, à 1024 m, exactement sa performance
de 256 m ; la différence appariée (+0,02) est bien en dessous de la variabilité d'un épisode (M3), et le graphe du GNN n'y
change rien.*

## M2 — carte de voisinage vide (C4)

| Portée | modèle tel quel | G0+C4 | G1+C4 |
|---|---|---|---|
| 128 m | 1,98 ± 0,46 | 2,82 ± 0,62 | 2,82 ± 0,62 |
| 256 m | 1,69 ± 0,18 | 2,42 ± 0,48 | 2,48 ± 0,41 |
| 1024 m | 5,22 ± 1,41 | 3,20 ± 0,57 | 2,65 ± 0,44 |

*Conclusion M2 : la carte de voisinage est utile (sans elle, +0,8 à 128 m et +0,7 à 256 m) et c'est son contenu hors
distribution qui nuit (à 1024 m, une carte vide fait mieux que la carte complète, 3,2 contre 5,2, mais moins bien que la
carte à l'échelle 256 m, 1,7) ; une fois la carte vide, le plafonnement du graphe redevient visible (3,20 → 2,65).*

## M3 — sensibilité aux positions initiales (1024 m, 5 répétitions par environnement, bruit gaussien σ = 1 m, graines fixées)

| Cellule | écart-type intra-environnement (moyenne / médiane / max sur 11 env.) | écart-type entre environnements (nominal) | moyenne |
|---|---|---|---|
| G0+C0 | 0,35 / 0,32 / 0,66 | 1,41 | 5,2 |
| G0+C3 | 0,086 / 0,077 / 0,195 | 0,19 | 1,7 |
| G1+C3 | 0,082 / 0,076 / 0,154 | 0,21 | 1,7 |

Le clairvoyant recalculé sur les départs perturbés varie de 0,26 % (écart-type relatif) : la normalisation est stable.
L'épisode nominal n'est pas particulier : il sort parfois de l'intervalle des 5 répétitions (env. 5, G0+C0 : 4,66 contre
5,44 à 6,09 ; env. 10, G0+C3 : 1,61 contre 1,69 à 1,88).
*Conclusion M3 : un déplacement d'un mètre des positions initiales change le coût moyen d'un épisode d'environ 7 %
(0,35 pour le modèle tel quel, 0,08 à 0,09 pour les cellules réparées). Entre deux conditions, une différence par
environnement inférieure à ~0,1 (cellules réparées) ou ~0,35 (modèle tel quel) est dans le bruit d'un épisode ; sur
la moyenne de 11 environnements, le seuil est de l'ordre de 0,03 et 0,1 respectivement. Les effets du facteur CARTE
(−2,9) sont très au-dessus ; ceux du facteur GRAPHE dans le factoriel (−0,1 à 1024 m) sont dedans.*

## Lecture d'ensemble

Le modèle publié se dégrade parce que sa carte de voisinage change d'échelle et de contenu avec la portée ; dessinée à
une échelle fixe de 256 m, il devient invariant à la portée au niveau de son optimum (1,7), sous les deux CVT (2,5 à 2,6),
sans réentraînement ni modification du graphe. La résolution de 128 m (C2) est trop fine : elle perd les voisins entre
128 et 256 m qui aident. Figure : `fig_scale256.png`.
