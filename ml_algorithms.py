
import time
import sys

def typing_print(text, delay=0.01):
    for char in str(text):
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def loading_animation(message="Loading"):
    sys.stdout.write(message)
    sys.stdout.flush()
    for _ in range(3):
        time.sleep(0.3)
        sys.stdout.write(".")
        sys.stdout.flush()
    typing_print("\n")

### Complete Implementation for Google Colab
# ---
# **Supervised Learning:** Linear Regression, Logistic Regression, Decision Tree, Random Forest,
# SVM, KNN, Naive Bayes
# **Unsupervised Learning:** K-Means, Hierarchical Clustering, DBSCAN, PCA, Autoencoder

# ## ■ Import Libraries
# Code:
# !pip install scikit-learn matplotlib seaborn tensorflow -q
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Supervised
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Unsupervised
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

# Utilities
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
 accuracy_score, classification_report, confusion_matrix,
 mean_squared_error, r2_score, silhouette_score
)
from sklearn.datasets import (
 load_iris, load_wine, make_classification,
 make_blobs, make_moons, make_regression
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
typing_print('■ All libraries loaded!')

# ---
# # ■ PART 1 — SUPERVISED LEARNING
# ---
# # n Supervised & Unsupervised Learning Algorithms
# Sushant Dadheech
# ku_id - Ku2407u814

# ## 1■■ Linear Regression
# Code:
# # ■■ Data ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
X_reg, y_reg = make_regression(n_samples=200, n_features=1, noise=20, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# # ■■ Train ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
lr = LinearRegression()
lr.fit(X_tr, y_tr)
y_pred = lr.predict(X_te)

# # ■■ Metrics ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
typing_print('=== Linear Regression ===')
print(f'Coefficient : {lr.coef_[0]:.4f}')
print(f'Intercept : {lr.intercept_:.4f}')
print(f'RMSE : {np.sqrt(mean_squared_error(y_te, y_pred)):.4f}')
print(f'R² Score : {r2_score(y_te, y_pred):.4f}')

# # ■■ Plot ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(X_te, y_te, color='steelblue', alpha=0.6, label='Actual')
axes[0].plot(X_te, y_pred, color='red', lw=2, label='Regression Line')
axes[0].set_title('Linear Regression — Fit', fontweight='bold')
axes[0].set_xlabel('Feature'); axes[0].set_ylabel('Target'); axes[0].legend()

axes[1].scatter(y_te, y_pred, color='coral', alpha=0.7, edgecolors='k', s=40)
axes[1].plot([y_te.min(), y_te.max()], [y_te.min(), y_te.max()], 'k--', lw=2)
axes[1].set_title('Actual vs Predicted', fontweight='bold')
axes[1].set_xlabel('Actual'); axes[1].set_ylabel('Predicted')
plt.tight_layout(); plt.show()

# ## 2■■ Logistic Regression
# Code:
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_train)
X_te_s = scaler.transform(X_test)

log_reg = LogisticRegression(max_iter=200, random_state=42)
log_reg.fit(X_tr_s, y_train)
y_pred = log_reg.predict(X_te_s)

typing_print('=== Logistic Regression ===')
print(f'Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%')
print(classification_report(y_test, y_pred, target_names=iris.target_names))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
 xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Logistic Regression — Confusion Matrix', fontweight='bold')
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.tight_layout(); plt.show()

# ## 3■■ Decision Tree
# Code:
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

typing_print('=== Decision Tree ===')
print(f'Accuracy : {accuracy_score(y_test, y_pred_dt)*100:.2f}%')
print(f'Tree Depth : {dt.get_depth()}')
print(f'Num Leaves : {dt.get_n_leaves()}')

fig, axes = plt.subplots(1, 2, figsize=(20, 6))
plot_tree(dt, feature_names=iris.feature_names,
 class_names=iris.target_names, filled=True, rounded=True,
 fontsize=9, ax=axes[0])
axes[0].set_title('Decision Tree Visualization', fontweight='bold')

feat_imp = pd.Series(dt.feature_importances_, index=iris.feature_names).sort_values()
feat_imp.plot(kind='barh', ax=axes[1], color='steelblue')
axes[1].set_title('Feature Importance', fontweight='bold')
plt.tight_layout(); plt.show()

# ## 4■■ Random Forest
# Code:
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

typing_print('=== Random Forest ===')
print(f'Accuracy: {accuracy_score(y_test, y_pred_rf)*100:.2f}%')
print(classification_report(y_test, y_pred_rf, target_names=iris.target_names))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
pd.Series(rf.feature_importances_, index=iris.feature_names).sort_values().plot(
 kind='barh', ax=axes[0], color='forestgreen')
axes[0].set_title('Feature Importance — Random Forest', fontweight='bold')

sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Greens',
 xticklabels=iris.target_names, yticklabels=iris.target_names, ax=axes[1])
axes[1].set_title('Confusion Matrix', fontweight='bold')
axes[1].set_ylabel('Actual'); axes[1].set_xlabel('Predicted')
plt.tight_layout(); plt.show()

# ## 5■■ Support Vector Machine (SVM)
# Code:
X_bin, y_bin = make_classification(n_samples=200, n_features=2, n_redundant=0,
 n_clusters_per_class=1, random_state=42)
X_tb, X_vb, y_tb, y_vb = train_test_split(X_bin, y_bin, test_size=0.2, random_state=42)

kernels = ['linear', 'rbf', 'poly']
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, kernel in zip(axes, kernels):
 svm = SVC(kernel=kernel, C=1.0, random_state=42)
 svm.fit(X_tb, y_tb)
 acc = accuracy_score(y_vb, svm.predict(X_vb))
 print(f'SVM {kernel:6s} | Accuracy: {acc*100:.2f}%')
 
 xx, yy = np.meshgrid(
 np.linspace(X_bin[:,0].min()-1, X_bin[:,0].max()+1, 200),
 np.linspace(X_bin[:,1].min()-1, X_bin[:,1].max()+1, 200)
 )
 Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
 ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
 ax.scatter(X_bin[:,0], X_bin[:,1], c=y_bin, cmap='coolwarm',
 edgecolors='k', s=30, linewidths=0.4)
 ax.set_title(f'SVM — {kernel.upper()} | Acc: {acc*100:.1f}%', fontweight='bold')

plt.suptitle('SVM Decision Boundaries', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()

# ## 6■■ K-Nearest Neighbors (KNN)
# Code:
k_scores = []
K_range = range(1, 21)
for k in K_range:
 knn = KNeighborsClassifier(n_neighbors=k)
 knn.fit(X_tr_s, y_train)
 k_scores.append(accuracy_score(y_test, knn.predict(X_te_s)))

best_k = K_range[np.argmax(k_scores)]
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_tr_s, y_train)
y_pred_knn = knn_best.predict(X_te_s)

typing_print('=== K-Nearest Neighbors ===')
print(f'Best K : {best_k}')
print(f'Accuracy : {accuracy_score(y_test, y_pred_knn)*100:.2f}%')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(K_range, k_scores, 'o-', color='purple', lw=2)
axes[0].axvline(x=best_k, color='red', linestyle='--', label=f'Best K={best_k}')
axes[0].set_title('KNN — Accuracy vs K', fontweight='bold')
axes[0].set_xlabel('K'); axes[0].set_ylabel('Accuracy'); axes[0].legend()

sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt='d', cmap='Purples',
 xticklabels=iris.target_names, yticklabels=iris.target_names, ax=axes[1])
axes[1].set_title(f'KNN (K={best_k}) — Confusion Matrix', fontweight='bold')
axes[1].set_ylabel('Actual'); axes[1].set_xlabel('Predicted')
plt.tight_layout(); plt.show()

# ## 7■■ Naive Bayes
# Code:
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

typing_print('=== Naive Bayes ===')
print(f'Accuracy: {accuracy_score(y_test, y_pred_nb)*100:.2f}%')
print(classification_report(y_test, y_pred_nb, target_names=iris.target_names))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_nb), annot=True, fmt='d', cmap='Oranges',
 xticklabels=iris.target_names, yticklabels=iris.target_names, ax=axes[0])
axes[0].set_title('Naive Bayes — Confusion Matrix', fontweight='bold')
axes[0].set_ylabel('Actual'); axes[0].set_xlabel('Predicted')

proba_df = pd.DataFrame(nb.predict_proba(X_test[:10]), columns=iris.target_names)
proba_df.plot(kind='bar', ax=axes[1], colormap='tab10', alpha=0.85)
axes[1].set_title('Prediction Probabilities — First 10 Samples', fontweight='bold')
axes[1].set_xlabel('Sample'); axes[1].set_ylabel('Probability')
axes[1].set_xticklabels(range(10), rotation=0)
plt.tight_layout(); plt.show()

# ## ■ Supervised — Model Comparison
# Code:
models = {
 'Logistic Regression': LogisticRegression(max_iter=200, random_state=42),
 'Decision Tree' : DecisionTreeClassifier(max_depth=4, random_state=42),
 'Random Forest' : RandomForestClassifier(n_estimators=100, random_state=42),
 'SVM (RBF)' : SVC(kernel='rbf', random_state=42),
 'KNN' : KNeighborsClassifier(n_neighbors=best_k),
 'Naive Bayes' : GaussianNB()
}
results = {}

typing_print('=== Model Comparison ===')
for name, model in models.items():
 model.fit(X_tr_s, y_train)
 acc = accuracy_score(y_test, model.predict(X_te_s))
 results[name] = round(acc * 100, 2)
 print(f'{name:25s}: {acc*100:.2f}%')

results_s = pd.Series(results).sort_values()
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(results_s)))

plt.figure(figsize=(10, 6))
bars = plt.barh(results_s.index, results_s.values, color=colors, edgecolor='black')
for bar, val in zip(bars, results_s.values):
 plt.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
 f'{val:.1f}%', va='center', fontweight='bold')
plt.xlim(80, 105)
plt.title('Supervised Learning — Accuracy Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Accuracy (%)')
plt.tight_layout(); plt.show()

# ---
# # ■ PART 2 — UNSUPERVISED LEARNING
# ---

# ## 1■■ K-Means Clustering
# Code:
X_blob, _ = make_blobs(n_samples=400, centers=4, cluster_std=0.8, random_state=42)

# # Elbow + Silhouette
inertia, sil_scores = [], []
K_range = range(2, 11)
for k in K_range:
 km = KMeans(n_clusters=k, random_state=42, n_init=10)
 lbls = km.fit_predict(X_blob)
 inertia.append(km.inertia_)
 sil_scores.append(silhouette_score(X_blob, lbls))

best_k_km = K_range[np.argmax(sil_scores)]
kmeans = KMeans(n_clusters=best_k_km, random_state=42, n_init=10)
km_labels = kmeans.fit_predict(X_blob)

typing_print('=== K-Means Clustering ===')
print(f'Best K : {best_k_km}')
print(f'Silhouette Score : {silhouette_score(X_blob, km_labels):.4f}')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(K_range, inertia, 'bo-', lw=2); axes[0].set_title('Elbow Method', fontweight='bold')
axes[0].set_xlabel('K'); axes[0].set_ylabel('Inertia')

axes[1].plot(K_range, sil_scores, 'ro-', lw=2)
axes[1].axvline(x=best_k_km, color='blue', linestyle='--', label=f'Best K={best_k_km}')
axes[1].set_title('Silhouette Score vs K', fontweight='bold')
axes[1].set_xlabel('K'); axes[1].legend()

axes[2].scatter(X_blob[:,0], X_blob[:,1], c=km_labels, cmap='tab10', alpha=0.7,
 edgecolors='k', linewidths=0.3, s=40)
axes[2].scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
 marker='*', s=300, c='red', label='Centroids', zorder=5)
axes[2].set_title(f'K-Means (K={best_k_km})', fontweight='bold'); axes[2].legend()
plt.tight_layout(); plt.show()

# ## 2■■ Hierarchical Clustering
# Code:
X_hc = X_blob[:80] # subset for dendrogram
link_mat = linkage(X_hc, method='ward')

hc = AgglomerativeClustering(n_clusters=4, linkage='ward')
hc_labels = hc.fit_predict(X_blob)

typing_print('=== Hierarchical Clustering ===')
print(f'Silhouette Score: {silhouette_score(X_blob, hc_labels):.4f}')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
dendrogram(link_mat, ax=axes[0], truncate_mode='lastp', p=15,
 leaf_rotation=45, leaf_font_size=10, color_threshold=6)
axes[0].set_title('Dendrogram', fontweight='bold')
axes[0].set_xlabel('Sample / Cluster'); axes[0].set_ylabel('Distance')

axes[1].scatter(X_blob[:,0], X_blob[:,1], c=hc_labels, cmap='tab10',
 alpha=0.7, edgecolors='k', linewidths=0.3, s=40)
axes[1].set_title('Hierarchical Clustering Result', fontweight='bold')
plt.tight_layout(); plt.show()

# ## 3■■ DBSCAN
# Code:
X_moon, _ = make_moons(n_samples=300, noise=0.1, random_state=42)
km_moon = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(X_moon)

db = DBSCAN(eps=0.2, min_samples=5)
db_labels = db.fit_predict(X_moon)

n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise = list(db_labels).count(-1)

typing_print('=== DBSCAN ===')
print(f'Clusters found : {n_clusters}')
print(f'Noise points : {n_noise}')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(X_moon[:,0], X_moon[:,1], c=km_moon, cmap='bwr',
 alpha=0.7, edgecolors='k', linewidths=0.3, s=40)
axes[0].set_title('K-Means on Moon Data (■ Fails)', fontweight='bold')

colors_db = ['gray' if l==-1 else plt.cm.tab10(l) for l in db_labels]
axes[1].scatter(X_moon[:,0], X_moon[:,1], c=colors_db,
 alpha=0.7, edgecolors='k', linewidths=0.3, s=40)
axes[1].set_title(f'DBSCAN (■ Works) — Clusters={n_clusters}, Noise={n_noise}', fontweight='bold')
plt.tight_layout(); plt.show()

# ## 4■■ Principal Component Analysis (PCA)
# Code:
wine = load_wine()
X_wine = StandardScaler().fit_transform(wine.data)

pca_full = PCA(random_state=42).fit(X_wine)
cumvar = np.cumsum(pca_full.explained_variance_ratio_) * 100

pca_2d = PCA(n_components=2, random_state=42)
X_pca = pca_2d.fit_transform(X_wine)

typing_print('=== PCA ===')
print(f'Original features : {X_wine.shape[1]}')
print(f'PC1 variance : {pca_full.explained_variance_ratio_[0]*100:.2f}%')
print(f'PC2 variance : {pca_full.explained_variance_ratio_[1]*100:.2f}%')
print(f'Components for 95% : {np.argmax(cumvar >= 95) + 1}')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].bar(range(1, len(pca_full.explained_variance_ratio_)+1),
 pca_full.explained_variance_ratio_*100, color='steelblue', alpha=0.7)
axes[0].set_title('Scree Plot', fontweight='bold')
axes[0].set_xlabel('Principal Component'); axes[0].set_ylabel('Explained Variance (%)')

axes[1].plot(range(1, len(cumvar)+1), cumvar, 'go-', lw=2)
axes[1].axhline(y=95, color='red', linestyle='--', label='95% threshold')
axes[1].set_title('Cumulative Explained Variance', fontweight='bold')
axes[1].set_xlabel('Components'); axes[1].set_ylabel('Variance (%)'); axes[1].legend()

for cls, name in enumerate(wine.target_names):
 mask = wine.target == cls
 axes[2].scatter(X_pca[mask,0], X_pca[mask,1], label=name, s=50, alpha=0.8,
 edgecolors='k', linewidths=0.3)
axes[2].set_title('PCA — 2D Projection (Wine)', fontweight='bold')
axes[2].set_xlabel(f'PC1 ({pca_full.explained_variance_ratio_[0]*100:.1f}%)')
axes[2].set_ylabel(f'PC2 ({pca_full.explained_variance_ratio_[1]*100:.1f}%)')
axes[2].legend()
plt.tight_layout(); plt.show()

# ## 5■■ Autoencoder (Deep Learning)
# Code:
X_ae = StandardScaler().fit_transform(wine.data).astype(np.float32)
input_dim, latent_dim = X_ae.shape[1], 2

# Build Autoencoder
inp = keras.Input(shape=(input_dim,))
enc = layers.Dense(8, activation='relu')(inp)
enc = layers.Dense(latent_dim, activation='relu')(enc)
dec = layers.Dense(8, activation='relu')(enc)
out = layers.Dense(input_dim, activation='linear')(dec)

autoencoder = keras.Model(inp, out, name='Autoencoder')
encoder = keras.Model(inp, enc, name='Encoder')

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

history = autoencoder.fit(X_ae, X_ae, epochs=100, batch_size=16,
 validation_split=0.2, verbose=0)

X_encoded = encoder.predict(X_ae)
X_reconstructed = autoencoder.predict(X_ae)
rec_errors = np.mean(np.square(X_ae - X_reconstructed), axis=1)

typing_print('=== Autoencoder ===')
print(f'Mean Reconstruction Error : {np.mean(rec_errors):.6f}')
print(f'Max Reconstruction Error : {np.max(rec_errors):.6f}')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(history.history['loss'], label='Train')
axes[0].plot(history.history['val_loss'], label='Val')
axes[0].set_title('Training Loss', fontweight='bold')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('MSE'); axes[0].legend()

for cls, name in enumerate(wine.target_names):
 mask = wine.target == cls
 axes[1].scatter(X_encoded[mask,0], X_encoded[mask,1], label=name, s=50,
 edgecolors='k', linewidths=0.3, alpha=0.8)
axes[1].set_title('Latent Space (2D)', fontweight='bold'); axes[1].legend()

for cls, name in enumerate(wine.target_names):
 axes[2].hist(rec_errors[wine.target==cls], bins=20, alpha=0.6, label=name)
axes[2].set_title('Reconstruction Error', fontweight='bold')
axes[2].set_xlabel('MSE'); axes[2].set_ylabel('Count'); axes[2].legend()
plt.tight_layout(); plt.show()

# ## ■ Final Summary
# Code:
summary = pd.DataFrame({
 'Category' : ['Supervised']*7 + ['Unsupervised']*5,
 'Algorithm': ['Linear Regression','Logistic Regression','Decision Tree',
 'Random Forest','SVM','KNN','Naive Bayes',
 'K-Means','Hierarchical','DBSCAN','PCA','Autoencoder'],
 'Task' : ['Regression','Classification','Classification',
 'Classification','Classification','Classification','Classification',
 'Clustering','Clustering','Clustering',
 'Dim. Reduction','Representation'],
 'Strength' : ['Interpretable','Probabilistic output','Visualizable rules',
 'High accuracy','High-dim data','Simple, no training','Fast & low data',
 'Scalable & fast','No K needed','Arbitrary shapes',
 'Linear & fast','Non-linear reduction']
})

print(summary.to_string(index=False))
typing_print('\n■ All algorithms implemented successfully!')
