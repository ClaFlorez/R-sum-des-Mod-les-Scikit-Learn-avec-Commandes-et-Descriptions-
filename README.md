## 🌟 Résumé des Modèles Scikit-Learn avec Commandes et Descriptions (FR)

---

### 📘 1. Classification

Prédit une **catégorie** ou une **étiquette** (ex : maladie / pas maladie).

**Modèles principaux :**

* `LogisticRegression` : Modèle simple et efficace pour les données linéairement séparables.

  * **Exemple d'utilisation :** Prédire si un email est "spam" ou "non-spam".
  * **Résultat attendu :** Une étiquette binaire (0 ou 1) pour chaque email.

* `RandomForestClassifier` : Forêt d'arbres de décision pour une précision robuste.

  * **Exemple :** Classification d'images (ex. chiffres manuscrits).
  * **Résultat attendu :** Classe prédite parmi plusieurs (0-9 pour MNIST).

* `KNeighborsClassifier` : Classe selon les voisins les plus proches.

  * **Exemple :** Prédire le genre d'un film basé sur ses caractéristiques.
  * **Résultat :** Catégorie majoritaire parmi les k plus proches voisins.

* `SVC` : Machine à vecteurs de support (kernel linéaire ou non).

  * **Exemple :** Séparer deux espèces de fleurs.
  * **Résultat :** Frontière de décision maximale entre classes.

* `MLPClassifier` : Perceptron multicouche, réseau de neurones de base.

  * **Exemple :** Reconnaissance de chiffres manuscrits.
  * **Résultat :** Prédiction de la classe avec activation non linéaire.

**Exemple de code :**

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

### 📑 2. Régression

Prédit une **valeur continue** (ex : revenu, prix d'une maison).

**Modèles principaux :**

* `LinearRegression` : Modèle linéaire simple.

  * **Exemple :** Prédire le prix d'une maison selon sa taille.
  * **Résultat :** Une valeur numérique continue.

* `RandomForestRegressor` : Version régression des forêts aléatoires.

  * **Exemple :** Prédire la température moyenne d'une région.
  * **Résultat :** Moyenne pondérée des prédictions de tous les arbres.

* `Lasso` : Régression avec régularisation L1 (sélectionne les variables).

  * **Exemple :** Prédire le salaire avec sélection automatique des variables utiles.
  * **Résultat :** Régression linéaire simplifiée avec certaines variables mises à zéro.

* `SVR` : Régression par vecteurs de support.

  * **Exemple :** Prédire la demande d'énergie horaire.
  * **Résultat :** Régression avec marge d'erreur tolérée.

* `MLPRegressor` : Réseau de neurones pour régression.

  * **Exemple :** Prédire les ventes hebdomadaires d'un produit.
  * **Résultat :** Valeur continue non linéaire.

**Exemple de code :**

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

### 📃 3. Regroupement (Clustering)

Identifie automatiquement des **groupes** dans des données non étiquetées.

**Modèles principaux :**

* `KMeans` : Regroupe les points en k groupes basés sur la distance.

  * **Exemple :** Segmenter des clients selon leur comportement d’achat.
  * **Résultat :** Étiquettes de cluster (0, 1, 2...)

* `DBSCAN` : Densité de points pour identifier des clusters.

  * **Exemple :** Identifier des régions denses dans des données géographiques.
  * **Résultat :** Étiquettes de cluster ou -1 pour les bruits.

* `MeanShift` : Glissement de la moyenne vers des zones de forte densité.

  * **Exemple :** Détection de motifs dans des images.
  * **Résultat :** Nombre automatique de clusters basé sur la densité.

* `AgglomerativeClustering` : Approche hiérarchique ascendante.

  * **Exemple :** Grouper des documents similaires.
  * **Résultat :** Clusters hiérarchiques avec fusion progressive.

**Exemple de code :**

```python
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
model.fit(X)
labels = model.labels_
```

---

### 📊 4. Réduction de Dimension

Simplifie les données en **réduisant le nombre de variables**, tout en gardant les informations importantes.

**Modèles principaux :**

* `PCA` : Analyse en composantes principales.

  * **Exemple :** Visualisation 2D de données à 100 dimensions.
  * **Résultat :** Nouvelles variables principales expliquant la variance.

* `TruncatedSVD` : SVD pour données clairsemées (sparse).

  * **Exemple :** Réduction de matrices TF-IDF de documents.
  * **Résultat :** Projection dans un espace de dimension réduite.

* `TSNE` : Réduction non linéaire pour visualisation.

  * **Exemple :** Visualisation 2D de clusters.
  * **Résultat :** Carte 2D montrant la proximité locale entre points.

**Exemple de code :**

```python
from sklearn.decomposition import PCA
model = PCA(n_components=2)
X_reduced = model.fit_transform(X)
```

---

### 📄 5. Méthodes d'Ensemble

Combine plusieurs modèles simples pour **renforcer la précision**.

**Modèles principaux :**

* `BaggingClassifier` : Moyenne de plusieurs modèles sur différents sous-ensembles.

  * **Exemple :** Classification de clients avec instabilités (petits datasets).
  * **Résultat :** Meilleure stabilité et moins d’overfitting.

* `AdaBoostClassifier` : Apprentissage adaptatif avec des poids.

  * **Exemple :** Classification binaire avec exemples difficiles pondérés.
  * **Résultat :** Focus sur les erreurs des modèles précédents.

* `GradientBoostingClassifier` : Boosting par gradient (puissant mais lent).

  * **Exemple :** Prédiction de défauts de crédit.
  * **Résultat :** Prédictions améliorées par corrections successives.

* `VotingClassifier` : Vote majoritaire de plusieurs modèles.

  * **Exemple :** Prédiction robuste en combinant SVM, arbre et régression.
  * **Résultat :** Classe choisie par la majorité.

* `StackingClassifier` : Combine les sorties de plusieurs modèles via un méta-modèle.

  * **Exemple :** Fusion de modèles faibles avec apprentissage supervisé.
  * **Résultat :** Modèle final entraîné sur les prédictions des autres.

**Exemple de code :**

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
```

---

### 🔹 Bon à savoir :

Tous les modèles Scikit-Learn suivent ce schéma :

```python
model.fit(X_train, y_train)
model.predict(X_test)
model.score(X_test, y_test)  # optionnel
```

---

### 🧰 Structure Universelle Scikit-Learn – Exemple Complet

Voici un exemple complet que l'on peut adapter à presque tous les modèles (classification, régression, etc.) :

```python
# Étape 1 : Importation
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression  # Exemple, remplaçable
from sklearn.metrics import accuracy_score, classification_report

# Étape 2 : Chargement des données
from sklearn.datasets import load_iris
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Étape 3 : Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Étape 4 : Normalisation (facultatif mais recommandé)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Étape 5 : Entraînement du modèle
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Étape 6 : Prédictions
y_pred = model.predict(X_test_scaled)

# Étape 7 : Évaluation
print("Précision :", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=data.target_names))
```

> 💡 Pour les modèles de régression, utilisez `mean_squared_error`, `r2_score`, etc.
> 💡 Pour le clustering, pas besoin de `y`, utilisez simplement `.fit(X)` puis `.labels_`

---

Créé avec ❤️ pour les apprenants en IA
