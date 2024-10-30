import numpy as np
import pandas as pd
import glob
from sklearn.preprocessing import StandardScaler #type: ignore
from sklearn.manifold import TSNE #type: ignore
import umap #type: ignore
import pickle

def process_features(features_path, csv_path=None, scaler=None, reducer=None, apply_umap=True, apply_tsne=True):
    features_files = glob.glob(f"{features_path}/*.npy")
    show_f = [np.load(npy_file) for npy_file in features_files]
    features_med, youtube_ids, labels = [], [], []

    # Etiquetas (test no tenia)
    for i, feature in enumerate(show_f):
        if feature.size == 0:
            continue
        mean_row = np.mean(feature, axis=0)
        features_med.append(mean_row)
        youtube_id = features_files[i].split("\\")[-1].split("_")[0]
        youtube_ids.append(youtube_id)
        
        # Obtener etiquetas solo si hay CSV proporcionado
        if csv_path:
            csv_df = pd.read_csv(csv_path)
            label_row = csv_df[csv_df['youtube_id'] == youtube_id]
            label = label_row['label'].values[0] if not label_row.empty else None
            labels.append(label)

    features_array = np.array(features_med)
    
    # Escalar las características (entrenamiento ajusta un nuevo scaler)
    if scaler is None:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_array.astype(float))
    else:
        features_scaled = scaler.transform(features_array.astype(float))

    # UMAP y t-SNE -- posible un LDA o un PCA
    umap_data, tsne_data = None, None
    if apply_umap:
        if reducer is None:
            n_neighbors = min(10, features_scaled.shape[0] - 1)
            reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=2)
            umap_data = reducer.fit_transform(features_scaled)
        else:
            umap_data = reducer.transform(features_scaled)

    if apply_tsne:
        tsne = TSNE(n_components=2, random_state=42)
        tsne_data = tsne.fit_transform(features_scaled)

    # Casi listo para exportar
    youtube_ids_array = np.array(youtube_ids).reshape(-1, 1)
    labels_array = np.array(labels).reshape(-1, 1) if labels else None

    # Salidas
    umap_output = np.hstack((youtube_ids_array, labels_array, umap_data)) if labels else np.hstack((youtube_ids_array, umap_data))
    tsne_output = np.hstack((youtube_ids_array, labels_array, tsne_data)) if labels else np.hstack((youtube_ids_array, tsne_data))

    return umap_output, tsne_output, scaler, reducer

# 6 Guardar archivos para entrenamiento y prueba

# Procesar y guardar datos de entrenamiento
umap_train, tsne_train, scaler, reducer = process_features(
    features_path="features_response\\train_features\\r21d\\r2plus1d_34_32_ig65m_ft_kinetics",
    csv_path="csv\\train_subset_10.csv"
)
np.save("npy_umapL.npy", umap_train)
np.save("npy_tsneL.npy", tsne_train)

# Guardar el scaler y el reducer de UMAP
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("umap_reducer.pkl", "wb") as f:
    pickle.dump(reducer, f)

# Procesar y guardar datos de prueba
umap_test, tsne_test, _, _ = process_features(
    features_path="features_response\\test_features\\r21d\\r2plus1d_34_32_ig65m_ft_kinetics",
    scaler=scaler,
    reducer=reducer
)
np.save("npy_umap_test.npy", umap_test)
np.save("npy_tsne_test.npy", tsne_test)
