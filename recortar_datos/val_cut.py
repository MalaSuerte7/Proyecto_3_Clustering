import pandas as pd
import os

# Datos
## Leyendo para limpiar
df_10 = pd.read_csv("csv\\val_subset_10.csv")
df = pd.read_csv("csv\\val_subset.csv")
## Asignar y limpiar
ids_utiles = df_10['youtube_id']
ids_completas = df['youtube_id']
ids_inutiles = df[~df['youtube_id'].isin(df_10['youtube_id'])]

#print(len(ids_completas), " ", len(ids_utiles), " ", len(ids_inutiles))

# Eliminar de la base de datos
videos_ruta = "C:\\Users\\lifeg\\OneDrive\\Escritorio\\heavy_data\\val_subset"

for actl_video in  os.listdir(videos_ruta):
    v_youtube_id = actl_video.split('_')[0]

    if v_youtube_id in ids_inutiles.values:
        file_path = os.path.join(videos_ruta, actl_video)
        try:
            os.remove(file_path)
        except OSError as e:
            print("can't with: ", file_path)
# C:\\Users\\lifeg\\OneDrive\\Escritorio\\heavy_data\\train_subset