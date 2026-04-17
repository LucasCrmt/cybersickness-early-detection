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
| InceptionTime | Convolutif multi-échelle | Séries brutes | Reconnu dans la littérature, capture plusieurs échelles simultanément, disponible dans `tsai` (biblothèque python complémentaire à PyTorch) | Plus lourd que CNN 1D, nécessite parfois du transfer learning | Bon | Bon | Mauvais |
| BiLSTM | Récurrent bidirectionnel | Séries brutes | Modélise l'accumulation du malaise, contexte avant et après | Entraînement lent, vanishing gradient | Mauvais | Moyen | Mauvais |
| CNN-LSTM hybride | Convolutif + Récurrent | Séries brutes | Combine extraction locale et mémoire temporelle, référencé en littérature physiologique | Complexe à implémenter, risque d'overfitting | Mauvais | Moyen | Mauvais |


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



