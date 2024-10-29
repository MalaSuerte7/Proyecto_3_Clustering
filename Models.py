import numpy as np
import pandas as pd
from kmeans import kmeans_pp
from sklearn.metrics import silhouette_score, adjusted_rand_score, mutual_info_score # type: ignore


features_path = "C:\\Users\\lifeg\\OneDrive\\Escritorio\\Machine\\Proyecto_3_Clustering\\npy_umap.npy"
labels_path = "C:\\Users\\lifeg\\OneDrive\\Escritorio\\Machine\\Proyecto_3_Clustering\\npy_umapL.npy"

points = np.load(features_path)
l_points = np.load(labels_path)

l_points_pd = pd.DataFrame(l_points)
# print(l_points_pd.head())
l_points_filtered = l_points_pd.iloc[:, 1:4] 
# print(l_points_filtered.head())

# Kmean++ -----------------------------------------------------------
k_m = kmeans_pp(points, k=10)
k_m.fit()
# Silhuete ----------------------------------------------------------
sil_grade = silhouette_score(X= points, labels=k_m.assignment)
print("Silhuete: " , sil_grade)

# Rand Index (RI) ---------------------------------------------------
labels_true = l_points_pd[1].values
ri = adjusted_rand_score(labels_true=labels_true, labels_pred=k_m.assignment)
print("Rand Index: ", ri)

# Mutual Information (MI) -------------------------------------------
mi = mutual_info_score(labels_true=labels_true, labels_pred=k_m.assignment)
print("Mutual Information: ", mi)

