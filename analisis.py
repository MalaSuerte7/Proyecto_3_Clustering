import numpy as np
import pandas as pd
import glob
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.manifold import TSNE # type: ignore
import umap # type: ignore

# 1 Leer features 
test_features = "features_response\\train_features\\r21d\\r2plus1d_34_32_ig65m_ft_kinetics"
features_test = glob.glob(f"{test_features}/*.npy")
# 1.1 Leídos listos para mostrar
show_f_test = [np.load(npy_file) for npy_file in features_test] 
# 1.2 CSV con labels
csv_df = pd.read_csv("csv\\train_subset_10.csv") 

# 2 ERROR dimensional ---- Solución  
# 2.1 Analisando problemas de dimensionalidad
# for i, show_f_test in enumerate(show_f_test):
#     print(f"Matriz {i} forma: {show_f_test.shape}")

# 2.2 Tienen distintas filas, por lo que mejor redimensionar
# 2.3 Haremos un promedio para tener todas las filas [1,512]
features_med = []
youtube_ids = []
labels = []

for i, actual_feature in enumerate(show_f_test):
    if actual_feature.size == 0:
        continue
    
    # 2.3.1 youtube_id
    file = features_test[i]
    youtube_id = file.split("\\")[-1].split("_")[0] 
    # 2.3.2 label
    label_chance = csv_df[csv_df['youtube_id'] == youtube_id]
    if label_chance.empty:
        continue
    label = label_chance['label'].values[0]

    # 2.3.3 Promediar 
    mean_row = np.mean(actual_feature, axis=0)
    features_med.append(mean_row)
    youtube_ids.append(youtube_id)
    labels.append(label)

f_test = np.array(features_med)

# 2.5 Verificar solución   
# for i, f_test in enumerate(f_test):
#     print(f"Matriz {i} forma: {f_test.shape}")

# 3 Scalar
scaler = StandardScaler()
features_scaled = scaler.fit_transform(f_test.astype(float)) # Es el X

# 4 Metodos
# 4.1 Umaping
n_neighbors = min(10, features_scaled.shape[0] - 1) 
reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=2)
X_umap = reducer.fit_transform(features_scaled)

# 4.2 T-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(features_scaled) 
 
# ------------------------------------------------------------------------------------
# 5 Preparar datos de salida con youtube_id, label, UMAP feature 1 y UMAP feature 2
youtube_ids = np.array(youtube_ids).reshape(-1, 1)
labels = np.array(labels).reshape(-1, 1)
# 5.1 El npy de umap esta con y sin label
output_data_umapL = np.hstack((youtube_ids, labels, X_umap))
output_data_umap = X_umap
# 5.2 El npy de tsne 
output_data_tsneL = np.hstack((youtube_ids, labels, X_tsne))
output_data_tsne = X_tsne
# ------------------------------------------------------------------------------------

# 6 Guardar ambos archivos .npy
# 6.1 Umap
# 6.1.1 Numpys
npy_umapL = "npy_umapL.npy"
npy_umap = "npy_umap.npy"
np.save(npy_umapL, output_data_umapL)
np.save(npy_umap, output_data_umap)
# 6.1.1 Pickels
import pickle

with open("umap_reducer.pkl", "wb") as f:
    pickle.dump(reducer, f)

# 6.1 Tsne
npy_tsneL = "npy_tsneL.npy"
npy_tsne = "npy_tsne.npy"
np.save(npy_tsneL, output_data_tsneL)
np.save(npy_tsne, output_data_tsne)

# 7 Plot
# 7.1 Umap plot
# import umap.plot  #type: ignore
# import matplotlib.pyplot as plt #type: ignore

# umap.plot.points(reducer)
# plt.show()

# 7.2 Tsne plot


