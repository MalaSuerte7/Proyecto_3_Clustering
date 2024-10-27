import numpy as np
import pandas as pd
import glob
from sklearn.preprocessing import StandardScaler
import umap

# 1 Leer features 
#test_features = "features_response\\train_features\\r21d\\r2plus1d_34_32_ig65m_ft_kinetics"
test_features = "Act"
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
features_scaled = scaler.fit_transform(f_test[:, 2:].astype(float)) 

# 4 Umaping
n_neighbors = min(15, features_scaled.shape[0] - 1)  # Ajuste dinámico
reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=2)
features_umap = reducer.fit_transform(features_scaled)

# 5 Exportar para analisar y graficar
output_file = "features_umap1.npy"
np.save(output_file, features_umap)
