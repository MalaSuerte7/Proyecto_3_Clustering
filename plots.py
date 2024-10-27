import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Cargar las características UMAP y las etiquetas desde el archivo
features_umap = np.load(r"C:\Users\lifeg\OneDrive\Escritorio\Machine\Proyecto_3_Clustering\features_umap2.npy")

# Extraer los datos
youtube_ids = features_umap[:, 0]          # IDs de YouTube
labels = features_umap[:, 1]               # Etiquetas de clase
umap_features = features_umap[:, 2:].astype(float)  # Componentes de UMAP en 2D

################### Dispersion en 2D ##########################
# plt.figure(figsize=(10, 8))
# plt.scatter(umap_features[:, 0], umap_features[:, 1], s=5, alpha=0.6)
# plt.title('UMAP Projection of the Features')
# plt.xlabel('UMAP Component 1')
# plt.ylabel('UMAP Component 2')
# plt.grid()
# plt.show()

################### Trazado en 3D (opcional, pero con un tercer eje adicional) ##########################
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(umap_features[:, 0], umap_features[:, 1], zs=0, zdir='z', s=5, alpha=0.6)
ax.set_title('3D UMAP Projection of the Features')
ax.set_xlabel('UMAP Component 1')
ax.set_ylabel('UMAP Component 2')
ax.set_zlabel('UMAP Component 3 (Dummy axis)')
plt.show()

################### Clusters con etiquetas de clase ##########################
# Convertir etiquetas de texto a números si es necesario
# unique_labels, label_indices = np.unique(labels, return_inverse=True)

# plt.figure(figsize=(10, 8))
# scatter = plt.scatter(umap_features[:, 0], umap_features[:, 1], c=label_indices, s=5, cmap='viridis', alpha=0.6)
# plt.title('UMAP Projection with Clusters')
# plt.xlabel('UMAP Component 1')
# plt.ylabel('UMAP Component 2')
# plt.colorbar(scatter, label='Clusters')
# plt.grid()
# plt.show()
