import numpy as np
import pandas as pd
from kmeans import kmeans_pp
from sklearn.metrics import silhouette_score, adjusted_rand_score, mutual_info_score # type: ignore

# 1 Paths
# 1.1 Umap paths
umap_path = "C:\\Users\\lifeg\\OneDrive\\Escritorio\\Machine\\Proyecto_3_Clustering\\npy_umap.npy" 
umapL_path = "C:\\Users\\lifeg\\OneDrive\\Escritorio\\Machine\\Proyecto_3_Clustering\\npy_umapL.npy"
# 1.2 Tsne paths
tsne_path = "C:\\Users\\lifeg\\OneDrive\\Escritorio\\Machine\\Proyecto_3_Clustering\\npy_tsne.npy" 
tsneL_path = "C:\\Users\\lifeg\\OneDrive\\Escritorio\\Machine\\Proyecto_3_Clustering\\npy_tsneL.npy"

# 2 Load those
# 2.1 Umap loads
umap_points = np.load(umap_path)
umap_l_points = np.load(umapL_path)
umap_l_points_pd = pd.DataFrame(umap_l_points)
# print(l_points_pd.head())
# l_points_filtered = l_points_pd.iloc[:, 1:4] 
# print(l_points_filtered.head())

# 2.2 Tsne
tsne_points = np.load(tsne_path)
tsne_l_points = np.load(tsneL_path)
tsne_l_points_pd = pd.DataFrame(tsne_l_points)

############################### kmean++ y Umap ########################################
print("-------Resultados con Kmens++ y Umap-------")
# Kmean++ -----------------------------------------------------------
k_m = kmeans_pp(umap_points, k=10)
k_m.fit()

# Silhuete ----------------------------------------------------------
sil_grade = silhouette_score(X=umap_points, labels=k_m.assignment)
print("Silhuete: " , sil_grade) # 10 veces, promedio:0,7 aprox

# Rand Index (RI) ---------------------------------------------------
labels_true = umap_l_points_pd[1].values
ri = adjusted_rand_score(labels_true=labels_true, labels_pred=k_m.assignment)
print("Rand Index: ", ri) # 10 veces, promedio:0.8 aprox

# Mutual Information (MI) -------------------------------------------
mi = mutual_info_score(labels_true=labels_true, labels_pred=k_m.assignment)
print("Mutual Information: ", mi)
print()
############################### kmean++ y tsne ########################################
print("-------Resultados con Kmens++ y Tsne-------")
# Kmean++ -----------------------------------------------------------
k_m = kmeans_pp(tsne_points, k=10)
k_m.fit()

# Silhuete ----------------------------------------------------------
sil_grade = silhouette_score(X= tsne_points, labels=k_m.assignment)
print("Silhuete: " , sil_grade) # 10 veces, promedio:0,6 aprox

# Rand Index (RI) ---------------------------------------------------
labels_true = tsne_l_points_pd[1].values
ri = adjusted_rand_score(labels_true=labels_true, labels_pred=k_m.assignment)
print("Rand Index: ", ri) # 10 veces, promedio:0,8 aprox

# Mutual Information (MI) -------------------------------------------
mi = mutual_info_score(labels_true=labels_true, labels_pred=k_m.assignment)
print("Mutual Information: ", mi)