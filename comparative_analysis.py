"""
Comparative Analysis Practical
K-Means Clustering, Logistic Regression, and Random Forest on Breast Cancer Data.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, silhouette_score

# Load and clean data
Data = pd.read_csv("breast-cancer.csv")
Data = Data.dropna()
print(Data.head())

X = Data.drop(columns=['diagnosis','id','Unnamed: 32'], errors='ignore')
y = Data['diagnosis']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Unsupervised Learning: K-Means Clustering ---
Kmeans = KMeans(n_clusters=2, random_state=42)
cluster = Kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

sil_score = silhouette_score(X_scaled, cluster)

plt.figure(figsize=(10,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=cluster, cmap='viridis', s=50, alpha=0.8)
plt.title("Kmeans Clustering on Breast Cancer Data")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.colorbar(label='Cluster')
plt.show()

# --- Supervised Learning ---
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)
log_pred = log_reg.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, log_pred)
lr_f1 = f1_score(y_test, log_pred)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_preds = rf_model.predict(X_test_scaled)
rf_acc = accuracy_score(y_test, rf_preds)
rf_f1 = f1_score(y_test, rf_preds)

print("\n" + "="*40)
print(" MODEL COMPARISON SUMMARY")
print("="*40)
print(f"K-Means (Unsupervised)")
print(f" -> Silhouette Score: {sil_score:.4f} (Closer to 1 is better)")
print("-" * 40)
print(f"Logistic Regression (Supervised)")
print(f" -> Accuracy: {lr_acc:.4f}")
print(f" -> F1-Score: {lr_f1:.4f}")
print("-" * 40)
print(f"Random Forest (Supervised)")
print(f" -> Accuracy: {rf_acc:.4f}")
print(f" -> F1-Score: {rf_f1:.4f}")
print("="*40)

print("Justification:")
print("I used K-Means to explore the natural, unbiased groupings of the cells,\n"
      "Logistic Regression to establish a highly interpretable baseline,\n"
      "and Random Forest to capture complex, non-linear patterns\n"
      "and maximize predictive accuracy.")
