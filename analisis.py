import numpy as np
import pandas as pd
import glob
from sklearn.preprocessing import StandardScaler
import umap

# 1 Leer features 
test_features = "features_response\\test_features\\r21d\\r2plus1d_34_32_ig65m_ft_kinetics"
features_test = glob.glob(f"{test_features}/*.npy")
# 1.1 Leídos listos para mostrar
show_f_test = [np.load(npy_file) for npy_file in features_test]

# 2 ERROR dimensional ---- Solución
# 2.1 Analisando problemas de dimensionalidad
# for i, show_f_test in enumerate(show_f_test):
#     print(f"Matriz {i} forma: {show_f_test.shape}")

# 2.2 Tienen distintas filas, por lo que mejor redimensionar
# 2.3 Haremos un promedio para tener todas las filas [1,512]
features_med = []
for actual_feature in show_f_test:
    if actual_feature.size == 0:
        continue

    # 2.3.1 Promediar 
    mean_row = np.mean(actual_feature, axis=0) 
    features_med.append(mean_row)

f_test = np.array(features_med)

# 2.5 Verificar solución   
# for i, f_test in enumerate(f_test):
#     print(f"Matriz {i} forma: {f_test.shape}")

# 3 Scalar
scaler = StandardScaler()
features_scaled = scaler.fit_transform(f_test) 

# 4 Umaping
reducer = umap.UMAP()
umaping_features = umap.UMAP(n_components = 2)
features_umap = reducer.fit_transform(features_scaled)

# 5 Exportar para analisar y graficar
output_file = "features_umap.npy"
np.save(output_file, features_umap)