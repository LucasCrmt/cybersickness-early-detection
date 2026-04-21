# Cadre méthodologique du prototype

## Objectif général
Mettre en place un cadre méthodologique reproductible pour prédire le sentiment malaise à partir de données physiologiques, en comparant deux familles d’approches :

- **Approche A** : approche classique à partir d’indicateurs calculés et agrégés
- **Approche B** : approche par séries temporelles exploitant directement la dynamique des signaux

L’objectif du cadre est de standardiser :
- la définition des tâches
- le prétraitement
- la construction des représentations
- la comparaison des modèles
- l’évaluation 
- l’interprétation des résultats.

DISCLAIMER: l'ensemble de propositions d'indicateurs/représentations graphiques ne sont pas finalisés, Ils sont à revoir.

# 1. Définition des objectifs prédictifs

Le cadre doit permettre de tester plusieurs formulations de la tâche :

## 1.1 Tâches possibles
- **Classification binaire** : `malade` / `pas malade`
- **Classification multi-niveaux** : `faible`, `moyen`, `élevé`, éventuellement `très élevé`
- **Régression** : prédiction d’un score continu de malaise
- **Prédiction précoce** : prédire l’état futur à partir d’une portion initiale du signal

## 1.2 Implémentation prévue
Pour chaque expérimentation, il faudra définir explicitement :
- la variable cible (sortie)
- la granularité temporelle de la prédiction
- l’instant de prédiction
- l’unité d’analyse :
  - sujet,
  - session,
  - fenêtre temporelle,
  - tâche complète.

---

# 2. Prétraitement des données

Le prétraitement doit être commun dans sa logique, mais adapté à la nature des représentations utilisées en A et B.

## 2.1 Étapes génériques
- suppression des séries ou essais non exploitables
- gestion des valeurs manquantes
- détection / atténuation des valeurs aberrantes
- harmonisation des unités
- normalisation ou standardisation.

## 2.2 Décisions méthodologiques
- suppression ou clipping des valeurs extrêmes
- normalisation :
  - globale
  - par sujet
  - par série
- sélection ou exclusion de certaines variables calculées 
- durée retenue pour l’analyse :
  - totalité de la tâche,
  - par minute,
  - par fenêtre glissante.

## 2.3 Implémentation prévue
Le cadre devra produire :
- une pipeline de prétraitement
- des statistiques avant / après nettoyage
- le nombre de sujets / essais exclus
- la justification des choix appliqués.

## 2.4 Représentations graphiques prévues

- A déterminer
---

# 3. Construction des représentations des données

C’est ici que la différence entre les approches A et B doit être explicitement structurée.

---

## 3.A Approche A - Représentations agrégées

### Principe
Les signaux sont résumés par des variables calculées, généralement agrégées sur l’ensemble de la tâche ou sur un segment donné.

### Représentations possibles
- moyenne
- variance / écart-type
- pente, dérive
- indices fréquentiels
- descripteurs physiologiques dérivés (Fourrier, etc...);
- statistiques résumées par phase ou par minute.

### Implémentation prévue
Pour chaque essai ou sujet :
- extraction des variables calculées ;
- assemblage sous forme de tableau tabulaire ;
- normalisation des features ;
- gestion des variables redondantes ou peu informatives.

### Questions méthodologiques traitées
- quelles variables sont retenues
- quelles variables sont exclues
- intérêt présumé de chaque variable
- comparaison entre sous-ensembles de variables

### Représentations graphiques prévues
- matrice de corrélation
- boxplots des variables par classe
- violin plots par niveau de malaise (sous-groupe au sein de la distribution)
- importance des variables
- projection PCA ou UMAP pour visualiser la séparabilité (projection 2D linéaire / non linéaire).

---

## 3.B Approche B - Représentations temporelles

### Principe
Les modèles reçoivent directement les séries temporelles, sans réduction à quelques indicateurs globaux.

### Représentations possibles
- signal brut normalisé
- fenêtres temporelles fixes
- fenêtres glissantes (0-10 s, 5-15 s etc...)
- séquences multimodales synchronisées ?
- représentation mono- ou multi-variable (eye-tracking seulement, tous, eye + head).

### Paramètres à fixer
- longueur de fenêtre
- pas de déplacement
- recouvrement
- horizon de prédiction (où se situe la prédiction par rapport à la fênetre utilisé)
- nombre de variables temporelles retenues.

### Implémentation prévue
- segmentation en séquences
- alignement temporel
- normalisation par fenêtre ou par série
- conservation des métadonnées :
  - sujet,
  - condition,
  - position temporelle,
  - durée originale.

### Questions méthodologiques traitées
- quelle durée de séquence est la plus informative
- quelle modalité apporte le plus
- la dynamique temporelle impacte-t-elle réellement la prédiction

### Représentations graphiques prévues
- affichage de séquences exemples
- superposition de signaux par classe
- visualisation de fenêtres temporelles
- distribution du nombre de fenêtres par classe
- cartes de saillance ou importance temporelle.

---

# 4. Choix du split train / validation / test

## 4.1 Principes
Le split doit être strictement défini pour éviter toute fuite d’information.

## 4.2 Stratégies envisagées
- split aléatoire stratifié
- split inter-individu avec séparation stricte des sujets
- éventuellement validation croisée selon la taille du dataset

## 4.3 Implémentation prévue
Le cadre devra systématiquement enregistrer :
- la méthode de split
- la seed utilisée
- la répartition des classes dans chaque sous-ensemble
- le nombre de sujets par sous-ensemble.

## 4.4 Représentations graphiques prévues
- tableau de répartition train / val / test ;
- barplots des effectifs par classe et par split
- répartition des sujets selon les ensembles.

---

# 5. Définition des modèles et des configurations


## 5.A Modèles pour l’approche A
(voir Tableau model_comparaison.md)

### Paramètres à documenter
- hyperparamètres principaux
- stratégie de pondération des classes (déséquilibre classe)
- prétraitements spécifiques (normalisation, etc...)
- sous-ensemble de variables utilisé

---

## 5.B Modèles pour l’approche B
(voir Tableau model_comparaison.md)

### Paramètres à documenter
- longueur des séquences
- nombre de couches
- kernel sizes
- unités LSTM
- dropout
- learning rate
- batch size.
...
---

# 6. Optimisation des hyperparamètres

## 6.1 Principe
Les modèles doivent être comparés dans des conditions homogènes.

## 6.2 Implémentation prévue
- définition d’un espace de recherche par modèle
- sélection sur ensemble de validation
- conservation des meilleures configurations

## 6.3 Sorties attendues
- tableau comparatif des configurations testées
- score de validation pour chaque configuration
- meilleure configuration retenue.

## 6.4 Représentations graphiques prévues
- tableau trié des résultats d’optimisation
- heatmap de performances selon les hyperparamètres
- courbes comparatives si nécessaire.

---

# 7. Entraînement reproductible

## 7.1 Règles communes
- seed fixée
- protocole identique entre expériences comparables
- callbacks standardisés
- traçabilité des paramètres.

## 7.2 Implémentation prévue
- enregistrement des paramètres d’entraînement
- sauvegarde des modèles finaux
- sauvegarde des historiques d’entraînement
- reproductibilité des splits et du prétraitement

## 7.3 Représentations graphiques prévues
Surtout pour l’approche B :
- courbes de loss
- courbes d’accuracy
- éventuellement courbes MAE/RMSE pour la régression

---

# 8. Évaluation des performances

## 8.1 Métriques classiques

### Pour la classification
- accuracy
- precision
- recall
- F1-score
- matrice de confusion.

### Pour la régression
- MAE
- RMSE
- R²
- corrélation prédiction / vérité.

## 8.2 Métriques clés pour le domaine physiologique
- généralisation inter-individu
- stabilité des prédictions
- robustesse au bruit ou aux perturbations
- sensibilité aux choix de séquençage.

## 8.3 Implémentation prévue
Le cadre devra produire :
- performances globales
- performances par sujet
- performances par classe
- performances selon le niveau de bruit ou la modalité utilisée.

## 8.4 Représentations graphiques prévues
- matrice de confusion
- barplots des métriques
- boxplots des scores par sujet
- courbes de robustesse sous bruit
- scatter plot vérité / prédiction en régression
- calibration ou distribution des probabilités si utile.

# 9. Analyse des patterns appris

---

## 9.A Pour l’approche A
L’objectif est d’identifier les variables les plus utiles.

### Implémentation prévue
- importance des variables
- permutation importance
- analyse de contribution locale si nécessaire.

### Représentations graphiques prévues
- barplot d’importance des variables
- top variables par modèle
- relation entre variable et prédiction.

## 9.B Pour l’approche B
L’objectif est d’identifier les portions temporelles ou motifs exploités par le modèle.

### Implémentation prévue
- cartes de saillance
- visualisation des segments les plus contributifs
- comparaison entre signaux correctement et incorrectement classés.

### Représentations graphiques prévues
- série temporelle + carte de saillance
- visualisation de séquences représentatives
- comparaison vrai label / label prédit
- analyse qualitative des motifs détectés.

---

# 10. Synthèse et interprétation des résultats

## 10.1 Synthèse attendue pour chaque modèle
Le cadre devra produire, pour chaque modèle étudié, une synthèse claire permettant d’expliquer sa pertinence méthodologique et ses limites dans le contexte de l’étude.  
Cette synthèse devra permettre de répondre aux questions suivantes :

- quel est l’objectif exact confié au modèle ?
- sur quelle représentation des données repose-t-il ?
  - approche A : variables calculées et agrégées ;
  - approche B : séries temporelles ou fenêtres temporelles ;
- quels choix de prétraitement et de séquençage ont été appliqués ?
- quels sont les hyperparamètres ou réglages retenus ?
- quelles performances globales le modèle obtient-il ?
- quelle est sa capacité de généralisation inter-individu ?
- quelle est sa stabilité face aux variations des données ?
- quelle est sa robustesse au bruit ou aux perturbations ?
- quels patterns ou variables semblent les plus importants dans sa décision ?
- quelles sont ses principales limites méthodologiques ?

## 10.2 Sorties finales attendues pour chaque modèle
Pour chaque modèle, le cadre devra fournir :

- une fiche synthétique du modèle
- le rappel de la tâche étudiée
- le type de représentation utilisé
- les choix de prétraitement retenus
- les paramètres et hyperparamètres principaux
- les métriques de performance
- les résultats de généralisation inter-individu
- les résultats de robustesse et de stabilité
- les visualisations d’interprétation
- une conclusion spécifique au modèle.

## 10.3 Représentations graphiques finales
Pour chaque modèle, les sorties graphiques finales pourront inclure :

- un tableau récapitulatif des paramètres et performances
- une matrice de confusion si la tâche est une classification
- un scatter plot vérité / prédiction si la tâche est une régression
- des boxplots de performance par sujet
- une courbe de robustesse sous bruit
- un graphique d’importance des variables pour l’approche A
- une visualisation de saillance (importance temporelle) pour l’approche B
- un schéma synthétique du pipeline appliqué au modèle ?

---

# Structure recommandée du cadre

# A déterminer