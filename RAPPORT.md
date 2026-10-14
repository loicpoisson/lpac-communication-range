# Rapport de session — l'anomalie de portée de LPAC se reproduit-elle avec le modèle publié ?

Session du 12 octobre 2026. Aucun entraînement. Modèle, environnements et paramètres : archive
`lpac_CoverageControl.tar.gz` de la release v1.3.1 de KumarRobotics/CoverageControl
(`model_k3_1024_state_dict.pt`, environnements 0 à 10, `coverage_control_params.toml`).
Métrique : coût de couverture moyen sur l'épisode (600 pas), divisé par celui de ClairvoyantCVT sur le même
environnement, comme dans les figures 15, 19 et 20 du papier. Moyennes et écarts-types sur les 11 environnements ;
valeurs par environnement dans `results.csv`.

## Verdict : **anomalie reproduite**, et bien au-delà du bruit entre environnements

| Portée (m) | LPAC-K3 | CVT décentralisé | CVT centralisé | degré moyen LPAC |
|---|---|---|---|---|
| 128 | 1,98 ± 0,46 | 2,59 ± 0,70 | 2,52 ± 0,74 | 1,4 |
| 256 | **1,69 ± 0,18** | 2,57 ± 0,71 | 2,52 ± 0,74 | 4,5 |
| 512 | 2,77 ± 0,38 | 2,59 ± 0,67 | 2,52 ± 0,74 | 11,7 |
| 768 | 4,52 ± 1,14 | 2,59 ± 0,67 | 2,52 ± 0,74 | 18,5 |
| 1024 | 5,22 ± 1,41 | 2,59 ± 0,67 | 2,52 ± 0,74 | 24,9 |

- La courbe de LPAC monte puis redescend, comme dans le papier, mais l'optimum est à **256 m** (pas 512) et la
  dégradation est massive : à 1024 m le coût triple par rapport à 256 m, et LPAC est **pire que les deux CVT dès 512 m**.
  À 768 et 1024 m le coût final dépasse même le coût initial (1,9 et 2,5 fois) : la politique détériore activement
  la couverture au lieu de l'améliorer.
- Elle est présente dans **chacun des 11 environnements** (ratio LPAC 1024 m / 256 m compris entre 2,1 et 4,7),
  très loin de l'écart-type entre environnements (0,18 à 256 m, 1,4 à 1024 m).
- Le papier compare des modèles réentraînés à chaque portée et observe une dégradation modérée ; ici un seul modèle,
  entraîné à 128 m, est poussé hors de son régime. L'ampleur n'est donc pas comparable à la figure 20 du papier ;
  la forme (montée puis descente) l'est.

## Étapes et où elles ont tourné

| Étape | Résultat | Où |
|---|---|---|
| 1. `eval_single` env 0, paramètres d'origine | LPAC 0,604 < Centralisé 0,611 < Décentralisé 0,701 (coût moyen / coût initial). Critère satisfait. 155 s. | WSL, CPU |
| 2. Balayage 5 portées × 11 env × 4 contrôleurs, 32 robots | complet, 220 lignes | WSL, CPU, 4 à 7 processus (`lpac_study.sweep`) |
| 3. Validation par les baselines | voir ci-dessous | |
| 4. Degré réel le long des trajectoires LPAC | 1,4 / 4,5 / 11,7 / 18,5 / 24,9 voisins (écart-type intra-épisode 1,1 à 5,5 ; 24 % de robots isolés à 128 m, 0 % dès 512 m) | mesuré sur l'`edge_index` fourni au GNN, `fig_degre.png` |
| 5. Balayage à degré constant | complet, 132 lignes | WSL |
| 6. Rang effectif des 5 couches TAGConv | mesuré à chaque pas, moyenné ; `fig_rang.png` | WSL |
| Notebook Colab `lpac_eval.ipynb` | exécuté localement de bout en bout en mode rapide (1 env, 60 pas) ; le balayage complet sur Colab n'a pas été lancé | à lancer par toi si besoin (3 à 5 h, surtout CPU) |

**Étape 3.** CentralizedCVT et ClairvoyantCVT ne lisent jamais la portée (vérifié dans le code C++) ; dans le balayage
WSL ils ont été calculés une fois par environnement et recopiés pour chaque portée (`computed = False` dans
`results.csv`), après vérification numérique sur l'env 0 que 128 m et 1024 m donnent des coûts identiques à 16 chiffres.
Le notebook Colab, lui, les recalcule à chaque portée. DecentralizedCVT : 2,594 → 2,567 → 2,591 → 2,591 → 2,591 ;
amélioration de 128 à 256 m, puis valeurs strictement identiques de 512 à 1024 m. C'est conforme au « s'améliore puis
stagne » du papier, et explicable : il calcule sa cellule de Voronoï dans une fenêtre locale de 256 m, donc un voisin
à plus de ~362 m ne peut plus la modifier. Le balayage n'est donc pas suspect.

**Étape 5 (degré ≈ 4 à 4,5 partout).**

| (portée, robots) | LPAC | CVT décentralisé | CVT centralisé | degré LPAC |
|---|---|---|---|---|
| (256, 32) | 1,69 ± 0,18 | 2,57 ± 0,71 | 2,52 ± 0,74 | 4,5 |
| (512, 11) | 1,84 ± 0,27 | 2,18 ± 0,38 | 2,08 ± 0,40 | 3,9 |
| (768, 7) | 1,74 ± 0,36 | 2,11 ± 0,39 | 2,05 ± 0,39 | 4,2 |
| (1024, 6) | 2,00 ± 0,57 | 2,02 ± 0,35 | 1,98 ± 0,35 | 4,5 |

À degré constant, la chute disparaît : LPAC reste entre 1,7 et 2,0 au lieu de monter à 5,2, et reste au niveau des
CVT ou mieux. Les équipes réduites utilisent les 6, 7 ou 11 premières positions des fichiers `.pos` (choix fixe,
sans graine, documenté dans `simulation.py`) avec le même champ d'importance. Deux défauts à garder en tête :
le nombre de robots change la tâche (les baselines bougent aussi, de 2,5 à 2,0), et à 6 robots le graphe à 1024 m
est **complet** (degré 5 = N − 1) : ce qui est maintenu constant est le nombre de voisins par robot, pas la fraction
d'arêtes. De plus l'hypothèse H1 du papier (collisions de robots dans la carte de voisinage 32 × 32) dépend elle aussi
du nombre de voisins, donc ce balayage ne la sépare pas de H3.

**Étape 6 (rang effectif, 32 robots, maximum 32).** Couche 2 : 22,5 (128 m) → 20,2 → 18,6 → 17,3 → 16,1 (1024 m) ;
couche 4 : 15,9 → 11,5 ; couche 5 : 8 à 9 à toutes les portées. Baisse graduelle et monotone avec la portée aux
couches 2 à 4, mais **aucun effondrement vers 1** : les représentations restent de rang élevé même sur un graphe à
25 voisins. L'ampleur de cette baisse (−30 %) ne ressemble pas à celle de la chute de performance (×3). Le sur-lissage
au sens strict n'est donc pas démontré ; les données sont compatibles avec un modèle recevant des entrées hors
distribution (degré, échelle de l'adjacence normalisée, cartes de voisinage saturées) sans que ses représentations
convergent.

## Mise en garde

Un seul modèle est publié, entraîné à 128 m. L'évaluer à 1024 m mélange deux effets : la densité du graphe et l'écart
au régime d'entraînement. Le balayage à degré constant sert à les séparer, et a son propre défaut : le nombre de robots
change la difficulté de la tâche. Les conclusions ci-dessus ne vont pas au-delà de ces données : l'anomalie existe
avec ce modèle, elle est liée au nombre de voisins plutôt qu'à la portée seule, mais la cause (sur-lissage, saturation
de la carte de voisinage, simple extrapolation) reste ouverte. Trancher demanderait des modèles entraînés à d'autres
portées, ce qui sort de cette session.

## Faits de code vérifiés (contredisent une partie du cadrage)

- `LPAC.forward` n'envoie pas `edge_weight` au GNN : les 5 `TAGConv` (K = 3) voient une adjacence **binaire**,
  normalisée symétriquement, sans boucles propres. Pas de somme, pas de GIN.
- Les cartes de communication sont calculées en C++ et s'échelonnent avec la portée (somme par cellule).
- L'évaluation est déterministe : l'aléa du simulateur ne sert qu'à générer des environnements et au bruit de
  position, désactivé. La configuration (256 m, 32 robots) calculée par deux processus différents a donné des
  valeurs identiques.
- Les évaluateurs du dépôt à HEAD sont incompatibles avec la roue 1.3.1 (`step()` renvoie un couple) : tout passe par
  le tag v1.3.1.

## Reproduire

`bash run_all.sh` (Linux/WSL, CPU, ~3 h sur 4 cœurs ; reprenable), ou `lpac_eval.ipynb` sur Colab. Détails dans `README.md`.
