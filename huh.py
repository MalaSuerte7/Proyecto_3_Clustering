import numpy as np
from kmeans import kmeans_pp
import sklearn.datasets as datasets
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score

# Load your data
features_path = r"C:\Users\lifeg\OneDrive\Escritorio\Machine\Proyecto_3_Clustering\features_umap_noLabel.npy"
labels_path = r"C:\Users\lifeg\OneDrive\Escritorio\Machine\Proyecto_3_Clustering\features_umap_label.npy"

data = np.load(features_path)
labels_data = np.load(labels_path)

# Extraer etiquetas de features_umap_label
labels = labels_data[:, 1]  # Columna de etiquetas

# Generate synthetic data based on the loaded centers
X, _ = datasets.make_blobs(n_samples=5409, centers=data, cluster_std=1, random_state=0)

# Initialize and fit kmeans++ model
kmeans = kmeans_pp(X, k=4)
kmeans.fit()

# Plot the clusters
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=kmeans.assignment)
plt.show()

# Calculate metrics
sil_score = silhouette_score(X, kmeans.assignment)
rand_index = adjusted_rand_score(_, kmeans.assignment)
mutual_info = adjusted_mutual_info_score(_, kmeans.assignment)

# Print metrics
print("Silhouette Score:", sil_score)
print("Adjusted Rand Index (RI):", rand_index)
print("Adjusted Mutual Information (MI):", mutual_info)

# Graficar los clusters
fig, ax = plt.subplots()
scatter = ax.scatter(data[:, 0], data[:, 1], c=kmeans.assignment, cmap='viridis')
plt.colorbar(scatter, ax=ax)
plt.show()