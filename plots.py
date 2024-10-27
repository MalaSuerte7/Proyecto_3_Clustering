import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# Cargar las características UMAP desde el archivo
features_umap = np.load(r"C:\Users\lifeg\OneDrive\Escritorio\Machine\Proyecto_3_Clustering\features_umap.npy")

################### Dispersion ##########################
# plt.figure(figsize=(10, 8))
# plt.scatter(features_umap[:, 0], features_umap[:, 1], s=5, alpha=0.6)
# plt.title('UMAP Projection of the Features')
# plt.xlabel('UMAP Component 1')
# plt.ylabel('UMAP Component 2')
# plt.grid()
# plt.show()
# # - puede ser
################### Contorno ##########################

# plt.figure(figsize=(10, 8))
# sns.kdeplot(x=features_umap[:, 0], y=features_umap[:, 1], fill=True, cmap='Blues', thresh=0, levels=100)
# plt.title('Contour Plot of UMAP Features')
# plt.xlabel('UMAP Component 1')
# plt.ylabel('UMAP Component 2')
# plt.show()
# # - nope

################### Trazado en 3D ##########################
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(features_umap[:, 0], features_umap[:, 1], zs=0, zdir='z', s=5, alpha=0.6)
# ax.set_title('3D UMAP Projection of the Features')
# ax.set_xlabel('UMAP Component 1')
# ax.set_ylabel('UMAP Component 2')
# ax.set_zlabel('UMAP Component 3')
# plt.show()
# # - sí, bonito 

################### Clusters ##########################
# # Supongamos que tienes una lista de etiquetas
# labels = np.random.randint(0, 2, size=features_umap.shape[0])  # Ejemplo de etiquetas aleatorias

# plt.figure(figsize=(10, 8))
# scatter = plt.scatter(features_umap[:, 0], features_umap[:, 1], c=labels, s=5, cmap='viridis', alpha=0.6)
# plt.title('UMAP Projection with Clusters')
# plt.xlabel('UMAP Component 1')
# plt.ylabel('UMAP Component 2')
# plt.colorbar(scatter, label='Clusters')
# plt.grid()
# plt.show()
# # - puede
