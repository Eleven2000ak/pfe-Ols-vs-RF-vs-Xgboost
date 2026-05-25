# Comparaison entre les méthodes statistiques classiques et les méthodes de Machine Learning pour la modélisation prédictive

## Régression linéaire (OLS) vs Random Forest vs XGBoost

---

**Encadrante** : Mme Kaouthar ELFASSI  
**Étudiant** : Abdesalam AKHDIM  
**PFE** – Modélisation prédictive

---

## 📌 Contexte et objectif

Ce projet compare trois approches de régression :

- Régression linéaire (OLS)
- Random Forest
- XGBoost

L’étude est réalisée sur le dataset *California Housing* afin de prédire le prix médian des logements.

Objectifs :

- comparer les performances prédictives
- analyser l’interprétabilité des modèles
- étudier la robustesse aux outliers
- évaluer la stabilité locale des prédictions

---

## 🧪 Hypothèses de recherche

| Hypothèse | Description | Résultat |
|---|---|---|
| $H1$ | Le processus génératif est linéaire. | Rejetée |
| $H2$ | Les modèles non linéaires améliorent les performances. | Validée |
| $H3$ | RF et XGBoost sont robustes aux outliers. | Validée |
| $H4$ | SHAP est fortement corrélé aux coefficients OLS. | Validée |

---

## 📊 Démarche méthodologique

### 1. Prétraitement

- chargement des données
- analyse exploratoire
- gestion des valeurs manquantes
- normalisation des variables

---

### 2. Entraînement des modèles

#### Régression linéaire (OLS)

Méthode des moindres carrés ordinaires :

$\hat{\beta}=
(X^TX)^{-1}X^Ty
$

Prédiction :

$\hat{y} = X\hat{\beta}$

---

#### Random Forest

Méthode ensembliste basée sur plusieurs arbres de décision.

- $100$ arbres
- réduction de variance par agrégation

Prédiction moyenne :


$\hat{y}(x)=
\frac{1}{B}
\sum_{b=1}^{B}
T_b(x)
$

---

#### XGBoost

Méthode de gradient boosting additive :

$
\hat{y}_i^{(t)}=
\hat{y}_i^{(t-1)}
+
f_t(x_i)
$

Fonction objectif :

$
\mathcal{L}^{(t)}=
\sum_{i=1}^{n}
l(y_i,\hat{y}_i^{(t)})
+
\Omega(f_t)
$

---

## 📈 Métriques d’évaluation

### Coefficient de détermination

$
R^2=
1 -
\frac{\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
{\sum_{i=1}^{n}(y_i-\bar{y})^2}
$

---

### Erreur absolue moyenne

$
MAE=
\frac{1}{n}
\sum_{i=1}^{n}
|y_i-\hat{y}_i|
$

---

### Racine de l’erreur quadratique moyenne

$
RMSE=
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
}
$

---

## 🧠 Interprétabilité – Valeurs SHAP

Valeur SHAP :

$
\phi_i(f,x)=
\sum_{S \subseteq F \setminus \{i\}}
\frac{|S|!(|F|-|S|-1)!}{|F|!}
\Big(
f(S \cup \{i\}) - f(S)
\Big)
$

Additivité :

$
f(x)=
\phi_0 + \sum_{i=1}^{p}\phi_i
$

---

## 🗺️ Local Decision Confidence Maps (LDCM)

Indice SDRloc :

$
\text{SDRloc}(i)=
1 -
\frac{
|\hat{y}_{RF}(i)-\hat{y}_{XGB}(i)|
}{
\max_j
|\hat{y}_{RF}(j)-\hat{y}_{XGB}(j)|
}
$

Interprétation :

- valeur proche de $1$ → forte confiance
- valeur proche de $0$ → faible confiance

---

## 📈 Résultats principaux

| Modèle | $R^2$ | MAE | Temps |
|---|---|---|---|
| OLS | 0.58 | 0.53 | < 0.1 s |
| Random Forest | 0.81 | 0.33 | 2.3 s |
| XGBoost | 0.83 | 0.31 | 1.8 s |

---

## 📌 Analyse des résultats

- XGBoost offre les meilleures performances globales.
- Random Forest est plus stable et robuste.
- OLS reste pertinent lorsque l’interprétabilité est prioritaire.

---

## 🧠 Grille décisionnelle

Fonction utilisée :

```python
def recommander_modele_avance(
    n_samples,
    linearite,
    besoin_interpretabilite,
    tolerance_risque='medium'
):
    if besoin_interpretabilite == 'élevé':
        return "Régression linéaire"

    elif linearite == 'faible' and n_samples > 5000:
        return "XGBoost" if tolerance_risque == 'high' else "Random Forest"

    else:
        return "Comparer OLS et RF"
```

---

## ✅ Conclusion générale

Cette étude montre que :

- les modèles non linéaires surpassent OLS sur données réelles
- SHAP améliore fortement l’interprétabilité
- les LDCM permettent une analyse spatiale de la confiance
- le choix du modèle dépend du compromis :
  - performance
  - robustesse
  - interprétabilité
  - coût calculatoire
