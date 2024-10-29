import pandas as pd
import numpy as np
import pickle
import umap #type: ignore
import umap.plot  #type: ignore
import matplotlib.pyplot as plt #type: ignore
import sklearn.datasets #type: ignore

X_umap = np.load("npy_umapL.npy", allow_pickle=True)
X_tsne = np.load("npy_tsne.npy", allow_pickle=True)

video_ids = X_umap[:, 0]  # ID de video
labels = X_umap[:, 1]  # Etiqueta de actividad
# Ploting Umap ---------------------------------------------------------------------

# Reducer
with open("umap_reducer.pkl", "rb") as f:
    reducer = pickle.load(f)

umap.plot.points(reducer)
umap.plot.points(reducer, labels=labels)

plt.show()








# x = features_umap[:, 2].astype(float)  # Coordenada X
# y = features_umap[:, 3].astype(float)  # Coordenada Y

# unique_labels = np.unique(labels)
# colors = plt.cm.get_cmap('tab10', len(unique_labels))
# label_to_color = {label: colors(i) for i, label in enumerate(unique_labels)}


# ##################### Dispersion en 2D ##########################
# plt.figure(figsize=(10, 7))

# # Graficar cada punto con su color correspondiente
# for i in range(len(x)):
#     plt.scatter(x[i], y[i], color=label_to_color[labels[i]], label=labels[i] if i == 0 else "")

# # Crear una leyenda con las etiquetas
# handles, labels_legend = [], []
# for label, color in label_to_color.items():
#     handles.append(plt.Line2D([0], [0], marker='o', color=color, markersize=5, linestyle=''))
#     labels_legend.append(label)

# plt.legend(handles, labels_legend, title="Activities")

# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('2D Scatter Plot of Features')
# plt.show()

