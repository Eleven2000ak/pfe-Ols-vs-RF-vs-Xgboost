# pfe-Ols-vs-RF-vs-Xgboost
# Comparaison entre les méthodes statistiques classiques et les méthodes de Machine Learning pour la modélisation prédictive

## Régression linéaire (OLS) vs Random Forest vs XGBoost

---

**Encadrante** : Mme Kaouthar ELFASSI  
**Étudiant** : Abdesalam AKHDIM  
**PFE** – Modélisation prédictive

---

## 📌 Contexte et objectif

Ce projet de fin d’études compare les performances et l’interprétabilité de trois modèles de régression :

- **Régression linéaire (OLS)** – méthode statistique classique
- **Random Forest** – méthode ensembliste non linéaire
- **XGBoost** – méthode de gradient boosting optimisée

L’application porte sur la prédiction des prix médians des logements en Californie (jeu de données *California Housing*). L’objectif est d’évaluer dans quels cas les modèles complexes surpassent l’approche linéaire, et comment interpréter leurs prédictions malgré leur complexité.

---

## 🧪 Hypothèses de recherche

| Hypothèse | Énoncé | Résultat obtenu |
|-----------|--------|----------------|
| **H1** | Le processus génératif des données est linéaire (ou au moins bien approximé par OLS). | **Rejetée** – OLS est significativement moins performant que RF/XGBoost. |
| **H2** | Sur des données réelles, les modèles non linéaires (RF, XGB) apportent un gain par rapport à OLS. | **Validée** – Les performances (R², MAE) sont nettement améliorées. |
| **H3** | RF et XGBoost sont robustes à la présence d’outliers. | **Validée** – Dégradation limitée après introduction de 5% d’outliers. |
| **H4** | L’importance des variables donnée par SHAP (XGBoost) est fortement corrélée avec les coefficients absolus de la régression linéaire. | **Validée** – Corrélation de Spearman ρ = 0.836, p‑value = 0.0013. |

---

## 📊 Démarche méthodologique

### 1. Prétraitement et exploration
- Chargement du jeu de données *California Housing*
- Analyse des distributions, corrélations
- Normalisation des variables (pour OLS)

### 2. Entraînement et évaluation
- **OLS** : régression linéaire classique (moindres carrés ordinaires)
- **Random Forest** : 100 arbres, critère MSE
- **XGBoost** : paramètres optimisés via grid search

**Métriques** : R², MAE, RMSE, temps d’entraînement.

### 3. Interprétabilité – Valeurs SHAP
- Calcul des valeurs SHAP avec `TreeExplainer` pour RF et XGB
- Importance globale : moyenne des |SHAP| par variable
- Graphiques *summary plot*, *bar plot*, *dependence plot*

### 4. Cartes de confiance locale (LDCM)
- Principe : utilisation du désaccord entre RF et XGB pour estimer la fiabilité spatiale
- Indice **SDRloc** (Spatial Disagreement Reliability) :
  \[
  \text{SDRloc}(i) = 1 - \frac{| \hat{y}_{RF}(i) - \hat{y}_{XGB}(i) |}{\max_j | \hat{y}_{RF}(j) - \hat{y}_{XGB}(j) |}
  \]
- Visualisation géographique des zones de confiance (vert = confiance élevée, rouge = faible)

### 5. Grille décisionnelle
- Fonction `recommander_modele_avance(n_samples, linearite, besoin_interpretabilite, tolerance_risque)`
- Logique conditionnelle pour choisir entre OLS, RF ou XGBoost selon le contexte métier

---

## 📈 Principaux résultats

| Modèle | R² (test) | MAE (test) | Temps d’entraînement |
|--------|-----------|------------|----------------------|
| OLS    | 0.58      | 0.53       | < 0.1 s              |
| Random Forest | 0.81 | 0.33 | 2.3 s |
| XGBoost        | 0.83 | 0.31 | 1.8 s |

**Conclusion** : XGBoost offre le meilleur compromis performance / temps. Random Forest est plus robuste aux réglages par défaut. OLS reste utile si l’interprétabilité est critique.

---

## 🗂️ Structure du dépôt
