# Rapport — par quel chemin le voisinage casse-t-il LPAC ? (plan factoriel à l'inférence)

Session du 12 octobre 2026. Modèle publié LPAC-K3 (entraîné à 128 m), 32 robots, 11 environnements de la
release, 600 pas, mêmes positions initiales que le balayage ; aucun poids modifié, aucun entraînement.
Le simulateur tourne toujours à la portée de test ; seules les **entrées du modèle** sont reconstruites :
le graphe du GNN (`edge_index`, facteur GRAPHE) et les deux canaux de carte de voisinage du CNN (facteur CARTE).
Coûts normalisés par le clairvoyant de `results.csv` (indépendant de la portée). Valeurs par environnement
dans `results_2x2.csv` (330 lignes).

**Conclusion : la carte de voisinage.** Le voisinage casse le modèle par les canaux donnés au CNN, pas par le
graphe donné aux couches TAGConv.

## Vérifications faites avant de mesurer

- Cartes de voisinage reconstruites en PyTorch à partir des positions (`neighbourhood.py`) : écart maximal à la
  bibliothèque **2,4 × 10⁻⁷** sur 30 pas à 1024 m et 10 pas à 128 m (`checks.py`). Canaux de `data.x` : 0 = carte
  d'importance locale, **1 et 2 = cartes de voisinage**, 3 = obstacles ; seuls 1 et 2 sont modifiés.
- Le graphe reconstruit coïncide avec l'`edge_index` de la bibliothèque et avec les listes de voisins du simulateur
  à chaque pas ; la portée simulée reste celle du test (1024 m pour tout le reste : capteurs, baselines).
- Non-régression : G0+C0 redonne **exactement** les lignes LPAC de `results.csv` (coût moyen identique à 10⁻¹⁵),
  à condition d'exécuter torch sur un seul thread ; en multi-thread l'écart est de 0,3 % sur le coût moyen, amplifié
  par la dynamique (6 % sur le coût final). Les 13 épisodes calculés à la fois sur WSL et sur FarmShare sont
  identiques bit à bit.
- G1 suit la règle k plus proches voisins de Muthusamy et al. (arXiv:2409.19829), où le graphe est **orienté** par
  définition (arête de chaque voisin j ∈ N_i vers i, adjacence binaire sans rayon). Ici le graphe est rendu
  **non orienté** (`to_undirected`), comme demandé et comme à l'entraînement, et restreint aux robots à portée.
  k = 3 est inclus dans la dose-réponse.

## Les six cases à 1024 m (k = 4), moyenne ± écart-type sur 11 environnements

| | C0 carte complète | C1 carte : k plus proches | C2 carte à l'échelle 128 m |
|---|---|---|---|
| **G0** graphe complet | **5,22 ± 1,41** (le modèle tel quel) | 2,91 ± 0,38 | 2,35 ± 0,53 |
| **G1** graphe plafonné à k | 5,11 ± 1,42 | 2,95 ± 0,58 | **2,18 ± 0,43** |

Degré réellement vu par le GNN : 24,9 (G0) contre 4,9 à 5,3 (G1). Robots dans la carte : 24,9 (C0), 5,2 (C1), 1,3 (C2).
Repères : LPAC à 256 m avec ses entrées normales 1,69 ± 0,18 ; CVT centralisé 2,52 ; CVT décentralisé 2,59.

**Ce que chaque facteur explique (effets principaux à 1024 m).** Passer de G0 à G1 change le coût de −0,11
(sous C0), +0,04 (C1) et −0,17 (C2) : rien, au regard d'un écart-type de 0,4 à 1,4 ; G1+C0 ne bat G0+C0 que dans 5
environnements sur 11. Passer de C0 à C1 change le coût de −2,3 (sous G0) et −2,2 (sous G1) ; de C0 à C2, de −2,9 et
−2,9. L'interaction est négligeable. Les cases C1 et C2 battent G0+C0 dans **11 environnements sur 11**. Le coût final
tombe de 10,9 (G0+C0) à 3,7 (C2) et 2,8 (G1+C2) : avec la carte réparée, la politique recommence à améliorer la
couverture au lieu de la détruire.

**Aux autres portées (k = 4).**

| Portée | G0+C0 | G0+C1 | G0+C2 | G1+C0 | G1+C1 | G1+C2 |
|---|---|---|---|---|---|---|
| 512 m | 2,77 ± 0,38 | 2,58 ± 0,51 | 2,26 ± 0,38 | 2,58 ± 0,31 | 2,51 ± 0,33 | 2,19 ± 0,45 |
| 768 m | 4,52 ± 1,14 | 3,03 ± 0,65 | 2,38 ± 0,43 | 4,27 ± 0,86 | 2,75 ± 0,46 | 2,18 ± 0,43 |

Même lecture : la carte explique presque tout ; à 512 m le graphe plafonné apporte un petit gain (2,77 → 2,58, dans
le bruit, 7 environnements sur 11), à 768 m rien de net.

## Dose-réponse à 1024 m (`fig_dose.png`)

| k | 2 | 3 | 4 | 8 | 16 | tous |
|---|---|---|---|---|---|---|
| G1+C1 (robots dans la carte) | 2,26 ± 0,21 (2,7) | 2,38 ± 0,19 (4,0) | 2,95 ± 0,58 (5,3) | 4,58 ± 1,33 (10,2) | 5,58 ± 1,44 (18,8) | 5,22 ± 1,41 (24,9) |
| G0+C1 | 2,36 ± 0,36 | 2,72 ± 0,30 | 2,91 ± 0,38 | 4,72 ± 1,17 | 5,75 ± 1,81 | 5,22 ± 1,41 |

La dégradation est monotone en nombre de robots dessinés dans la carte, et décroche entre 4 et 8 (c'est-à-dire
entre le degré d'entraînement à 256 m, 4,5, et le double). G0+C1 et G1+C1 se superposent : c'est bien le contenu de la
carte, pas le graphe, qui porte la dose.

## Le correctif sur les cinq portées (`fig_fix.png`, G1+C2, k = 4)

| Portée | 128 | 256 | 512 | 768 | 1024 |
|---|---|---|---|---|---|
| modèle tel quel | 1,98 ± 0,46 | **1,69 ± 0,18** | 2,77 ± 0,38 | 4,52 ± 1,14 | 5,22 ± 1,41 |
| G1+C2 | 1,99 ± 0,46 | 2,12 ± 0,23 | 2,19 ± 0,45 | 2,18 ± 0,43 | 2,18 ± 0,43 |
| G0+C2 | 1,98 ± 0,46 | 2,10 ± 0,29 | 2,26 ± 0,38 | 2,38 ± 0,43 | 2,35 ± 0,53 |

Avec la carte ramenée à l'échelle d'entraînement, la courbe ne redescend plus : le modèle devient pratiquement
**invariant à la portée** (à 128 m les entrées sont inchangées, d'où l'égalité ; à 768 et 1024 m, G1+C2 reçoit exactement
les mêmes entrées, d'où des coûts identiques) et reste sous les deux CVT partout. Mais il n'atteint plus son optimum
de 256 m (1,69) : les voisins supplémentaires de la carte à 256 m, dessinés à la résolution de 256 m, aident, et C2 les
supprime. Le correctif « à l'inférence » n'est donc pas une réparation complète : il borne la dégradation à ~2,2.

## Ce que ça dit, et ce que ça ne dit pas

- H3 (graphe trop dense, sur-lissage) est écartée pour ce modèle : plafonner le graphe à 5 voisins ne change rien,
  ce qui rejoint le rang effectif qui ne s'effondrait pas.
- H1 (carte de voisinage de résolution fixe) est confirmée comme chemin d'entrée. La dose-réponse ne distingue pas
  la perte d'information (plusieurs robots sommés par cellule) du simple décalage de distribution (plus de robots,
  normalisation /portée, cellules deux fois plus larges à chaque doublement) : C1 à k = 2 garde la résolution de
  1024 m et récupère presque autant que C2.
- Le degré 4 à 5 de G1 est aussi le degré d'entraînement : la non-réaction du GNN est établie pour un graphe
  plafonné à cette valeur, pas pour toute transformation du graphe.
- Mesure suivante proposée : dans C1, départager « nombre de robots » et « résolution » en gardant k = 2 à la
  résolution de 128 m, et à l'inverse tous les robots à portée mais avec une carte plus fine (64 × 64 redimensionnée),
  ce qui touche l'entrée du CNN sans réentraîner.

## Où ça a tourné

1024 m : WSL, CPU, 4 processus (~1 h 25). 768 m, 512 m, dose-réponse, 128/256 m : FarmShare (Slurm, 11 tâches par
lot, `farmshare_factorial.sbatch`, ~40 min au total). Reproduction : `python -m lpac_study.checks`,
`python -m lpac_study.factorial …`, `python -m lpac_study.plots_factorial …` (voir `README.md`), ou `run_all.sh`.
