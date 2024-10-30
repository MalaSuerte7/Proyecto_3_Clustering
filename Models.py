import numpy as np
import pandas as pd
from kmeans import kmeans_pp
from dbscan import DBSCAN
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

# 2.2 Tsne
tsne_points = np.load(tsne_path)
tsne_l_points = np.load(tsneL_path)
tsne_l_points_pd = pd.DataFrame(tsne_l_points)

# ############################### kmean++ y Umap ########################################
# print("-------Resultados con Kmens++ y Umap-------")
# # Kmean++ -----------------------------------------------------------
# k_mU = kmeans_pp(umap_points, k=10)
# k_mU.fit()

# # Silhuete ----------------------------------------------------------
# sil_grade = silhouette_score(X=umap_points, labels=k_mU.assignment)
# print("Silhuete: " , sil_grade) # 10 veces, promedio:0,7 aprox

# # Rand Index (RI) ---------------------------------------------------
# labels_true = umap_l_points_pd[1].values
# ri = adjusted_rand_score(labels_true=labels_true, labels_pred=k_mU.assignment)
# print("Rand Index: ", ri) # 10 veces, promedio:0.8 aprox

# # Mutual Information (MI) -------------------------------------------
# mi = mutual_info_score(labels_true=labels_true, labels_pred=k_mU.assignment)
# print("Mutual Information: ", mi)

# # Predicciones en el conjunto de prueba ------------------------------------------------
# umap_test_predictions = k_mU.predict(umap_points)
# umap_test_predictions_df = pd.DataFrame({
#     "youtube_id": [f"test_{i}" for i in range(len(umap_test_predictions))], #ids de otro lado
#     "cluster": umap_test_predictions
# })
# print("\nKmeans++ predictions for test data using UMAP:")
# print(umap_test_predictions_df.head())

# # Guardar predicciones de prueba en archivo CSV
# umap_test_predictions_df.to_csv("umap_test_predictions.csv", index=False)
# print("Test predictions using UMAP saved to 'umap_test_predictions.csv'.\n")

# print()
# ############################### kmean++ y tsne ########################################
# print("-------Resultados con Kmens++ y Tsne-------")
# # Kmean++ -----------------------------------------------------------
# k_mT = kmeans_pp(tsne_points, k=10)
# k_mT.fit()

# # Silhuete ----------------------------------------------------------
# sil_grade = silhouette_score(X= tsne_points, labels=k_mT.assignment)
# print("Silhuete: " , sil_grade) # 10 veces, promedio:0,6 aprox

# # Rand Index (RI) ---------------------------------------------------
# labels_true = tsne_l_points_pd[1].values
# ri = adjusted_rand_score(labels_true=labels_true, labels_pred=k_mT.assignment)
# print("Rand Index: ", ri) # 10 veces, promedio:0,8 aprox

# # Mutual Information (MI) -------------------------------------------
# mi = mutual_info_score(labels_true=labels_true, labels_pred=k_mT.assignment)
# print("Mutual Information: ", mi)

# # Predicciones en el conjunto de prueba ------------------------------------------------
# tsne_test_predictions = k_mT.predict(tsne_points)
# tsne_test_predictions_df = pd.DataFrame({
#     "youtube_id": [f"test_{i}" for i in range(len(tsne_test_predictions))],
#     "cluster": tsne_test_predictions
# })
# print("\nKmeans++ predictions for test data using t-SNE:")
# print(tsne_test_predictions_df.head())

# # Guardar predicciones de prueba en archivo CSV
# tsne_test_predictions_df.to_csv("tsne_test_predictions.csv", index=False)
# print("Test predictions using t-SNE saved to 'tsne_test_predictions.csv'.")


############################### DBSCAN y Umap ########################################
print("-------Resultados con DBSCAN y Umap-------")
# DBSCAN -----------------------------------------------------------
dbscan_umap = DBSCAN(data=umap_points, eps=0.5, min_points=5)  # Ajustar eps y min_points según sea necesario
umap_labels = dbscan_umap.fit()

# Silhouette --------------------------------------------------------
sil_grade = silhouette_score(X=umap_points, labels=umap_labels)
print("Silhouette: ", sil_grade)

# Rand Index (RI) ---------------------------------------------------
labels_true = umap_l_points_pd[1].values
ri = adjusted_rand_score(labels_true=labels_true, labels_pred=umap_labels)
print("Rand Index: ", ri)

# Mutual Information (MI) -------------------------------------------
mi = mutual_info_score(labels_true=labels_true, labels_pred=umap_labels)
print("Mutual Information: ", mi)

# Guardar etiquetas en archivo CSV
umap_dbscan_labels_df = pd.DataFrame({
    "youtube_id": [f"umap_{i}" for i in range(len(umap_labels))],
    "cluster": umap_labels
})
umap_dbscan_labels_df.to_csv("umap_dbscan_labels.csv", index=False)
print("DBSCAN labels using UMAP saved to 'umap_dbscan_labels.csv'.\n")

print()
############################### DBSCAN y tsne ########################################
print("-------Resultados con DBSCAN y Tsne-------")
# DBSCAN -----------------------------------------------------------
dbscan_tsne = DBSCAN(data=tsne_points, eps=0.5, min_points=5)  # Ajustar eps y min_points según sea necesario
tsne_labels = dbscan_tsne.fit()

# Silhouette --------------------------------------------------------
sil_grade = silhouette_score(X=tsne_points, labels=tsne_labels)
print("Silhouette: ", sil_grade)

# Rand Index (RI) ---------------------------------------------------
labels_true = tsne_l_points_pd[1].values
ri = adjusted_rand_score(labels_true=labels_true, labels_pred=tsne_labels)
print("Rand Index: ", ri)

# Mutual Information (MI) -------------------------------------------
mi = mutual_info_score(labels_true=labels_true, labels_pred=tsne_labels)
print("Mutual Information: ", mi)

# Guardar etiquetas en archivo CSV
tsne_dbscan_labels_df = pd.DataFrame({
    "youtube_id": [f"tsne_{i}" for i in range(len(tsne_labels))],
    "cluster": tsne_labels
})
tsne_dbscan_labels_df.to_csv("tsne_dbscan_labels.csv", index=False)
print("DBSCAN labels using t-SNE saved to 'tsne_dbscan_labels.csv'.")