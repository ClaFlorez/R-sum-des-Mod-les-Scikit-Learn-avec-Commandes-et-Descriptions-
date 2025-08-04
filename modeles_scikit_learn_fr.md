## 🌟 Résumé des Modèles Scikit-Learn avec Commandes et Descriptions (FR)

---

### 📘 1. Classification

Prédit une **catégorie** ou une **étiquette** (ex : maladie / pas maladie).

**Modèles principaux :**

- `` : Modèle simple et efficace pour les données linéairement séparables.
- `` : Forêt d'arbres de décision pour une précision robuste.
- `` : Classe selon les voisins les plus proches.
- `` : Machine à vecteurs de support (kernel linéaire ou non).
- `` : Perceptron multicouche, réseau de neurones de base.

**Exemple :**

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

- `` : Modèle linéaire simple.
- `` : Version régression des forêts aléatoires.
- `` : Régression avec régularisation L1 (sélectionne les variables).
- `` : Régression par vecteurs de support.
- `` : Réseau de neurones pour régression.

**Exemple :**

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

- `` : Regroupe les points en k groupes basés sur la distance.
- `` : Densité de points pour identifier des clusters.
- `` : Glissement de la moyenne vers des zones de forte densité.
- `` : Approche hiérarchique ascendante.

**Exemple :**

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

- `` : Analyse en composantes principales.
- `` : SVD pour données clairsemées (sparse).
- `` : Réduction non linéaire pour visualisation.

**Exemple :**

```python
from sklearn.decomposition import PCA
model = PCA(n_components=2)
X_reduced = model.fit_transform(X)
```

---

### 📄 5. Méthodes d'Ensemble

Combine plusieurs modèles simples pour **renforcer la précision**.

**Modèles principaux :**

- `` : Moyenne de plusieurs modèles sur différents sous-ensembles.
- `` : Apprentissage adaptatif avec des poids.
- `` : Boosting par gradient (puissant mais lent).
- `` : Vote majoritaire de plusieurs modèles.
- `` : Combine les sorties de plusieurs modèles via un méta-modèle.

**Exemple :**

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

Créé avec ❤️ pour les apprenants en IA

