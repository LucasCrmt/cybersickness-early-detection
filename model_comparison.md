# Comparaison des modèles

## Approche A: Indicateurs moyennés

| Modèle | Type | Entrée | Points forts | Limites | Robustesse N=42 | Robustesse N=1176 | Interprétabilité |
|---|---|---|---|---|---|---|---|
| Random Forest | Ensemble (arbres) | Features moyennées | Robuste petit dataset, feature importance, pas de normalisation requise | Ne capte pas la dynamique temporelle | Bon | Bon | Bon |
| XGBoost | Gradient boosting | Features moyennées | Très performant sur données tabulaires, gère les valeurs manquantes | Nombre important d'hyperparamètres, pas de séries temporelles | Bon | Bon | Bon |
| SVM | Marge de séparation | Features moyennées | Performant avec petit N, bonne généralisation (théorique) | Pas de feature importance, normalisation stricte requise | Bon | Bon | Mauvais |

## Approche B: Séries temporelles brutes

| Modèle | Type | Entrée | Points forts | Limites | Robustesse N=42 | Robustesse N=1176 | Interprétabilité |
|---|---|---|---|---|---|---|---|
| CNN 1D | Convolutif | Séries brutes | Rapide, détecte les patterns locaux, multivarié natif | Une seule échelle temporelle, pas de mémoire | Moyen | Bon | Moyen |
| InceptionTime | Convolutif multi-échelle | Séries brutes | SOTA benchmarks UCR/UEA, capture plusieurs échelles simultanément, disponible dans `tsai` | Plus lourd que CNN 1D, nécessite parfois du transfer learning | Bon | Bon | Mauvais |
| BiLSTM | Récurrent bidirectionnel | Séries brutes | Modélise l'accumulation du malaise, contexte avant et après | Entraînement lent, vanishing gradient | Mauvais | Moyen | Mauvais |
| CNN-LSTM hybride | Convolutif + Récurrent | Séries brutes | Combine extraction locale et mémoire temporelle, référencé en littérature physiologique | Complexe à implémenter, risque d'overfitting | Mauvais | Moyen | Mauvais |

---

## Approche C — Transformers

| Modèle | Type | Entrée | Points forts | Limites | Robustesse N=42 | Robustesse N=1176 | Interprétabilité |
|---|---|---|---|---|---|---|---|
| Temporal Fusion Transformer (TFT) | Transformer + attention temporelle | Séries brutes + covariables | Attention interprétable, gère les covariables statiques (profil participant) | Conçu pour la prédiction, très gourmand en données | Mauvais | Bon | Bon |
| PatchTST | Transformer par patches | Séries brutes | Efficace sur longues séquences (75 600 pts), SOTA 2023 | Sensible à la taille des patches, peu validé en biomécanique | Mauvais | Bon | Moyen |
| TimesNet | CNN 2D sur représentation temporelle | Séries brutes | Exploite les périodicités du signal (roulis/tangage), moins de paramètres qu'un Transformer pur | Architecture récente peu documentée, perd de l'info lors du passage 1D→2D | Mauvais | Moyen | Mauvais |

---

## Récapitulatif global

| Modèle | Approche | Complexité | Données requises | Capture dynamique temporelle | Robustesse N=42 | Robustesse N=1176 | Recommandation |
|---|---|---|---|---|---|---|---|
| Random Forest | A | Faible | Faible | Non | Bon | Bon | Baseline indispensable |
| XGBoost | A | Faible | Faible | Non | Bon | Bon | Baseline indispensable |
| SVM | A | Faible | Faible | Non | Bon | Bon | Complément utile en A |
| CNN 1D | B | Moyenne | Moyenne | Non | Moyen | Bon | Point de départ approche B |
| InceptionTime | B | Moyenne | Moyenne | Non | Bon | Bon | Meilleur choix approche B |
| BiLSTM | B | Élevée | Élevée | Oui | Mauvais | Moyen | Si progression temporelle clé |
| CNN-LSTM | B | Élevée | Élevée | Oui | Mauvais | Moyen | Exploratoire |
| TFT | C | Très élevée | Très élevée | Oui | Mauvais | Bon | Exploratoire, interprétable |
| PatchTST | C | Élevée | Très élevée | Oui | Mauvais | Bon | Exploratoire, SOTA 2023 |
| TimesNet | C | Élevée | Élevée | Partielle | Mauvais | Moyen | Si périodicités stables confirmées |

---

## Notes méthodologiques

- **Participants** : 42 sujets × 2 répétitions × 14 minutes = **1 176 fenêtres d'1 minute**
- **Validation** : Leave-One-Subject-Out (LOSO-CV) recommandée — regrouper les 28 fenêtres d'un même sujet (2 répétitions × 14 min) pour éviter toute fuite de données entre train et test
- **Data augmentation** : utile pour les approches B et C même avec N=1176, car les 1176 fenêtres restent issues de seulement 42 sujets indépendants
- **Fréquence d'échantillonnage** : 90 Hz → 5 400 points par fenêtre d'1 minute
- **Variable cible** : score d'inconfort de 0 à 20, renseigné toutes les minutes par le participant
- **Référence** : Pratviel Y., Laboratoire PSMS, Faculté des STAPS, URCA