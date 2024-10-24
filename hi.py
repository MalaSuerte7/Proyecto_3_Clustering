import os

# Ruta de la carpeta donde están los archivos .npy
npy_dir = "C:/Users/lifeg/OneDrive/Escritorio/Machine/Proyecto_3_Clustering/features_response/r21d/r2plus1d_34_32_ig65m_ft_kinetics"

# Obtener la lista de archivos en el directorio
files = os.listdir(npy_dir)

# Filtrar solo los archivos con extensión .npy
npy_files = [f for f in files if f.endswith('.npy')]

# Contar el número de archivos .npy
npy_count = len(npy_files)

print(f"Número de archivos .npy en la carpeta: {npy_count}")
import os

# Carpeta donde se guardan las características (archivos .npy)
output_dir = "C:/Users/lifeg/OneDrive/Escritorio/Machine/Proyecto_3_Clustering/features_response/r21d/r2plus1d_34_32_ig65m_ft_kinetics"

# Leer los archivos de características .npy ya existentes
npy_files = set(os.listdir(output_dir))

# Leer las rutas de los videos del archivo val_paths.txt
with open("C:/Users/lifeg/OneDrive/Escritorio/Machine/Proyecto_3_Clustering/train_paths.txt", "r") as f:
    video_paths = f.read().splitlines()

# Crear una lista filtrada con los videos que aún no se han procesado
videos_to_process = []

for video_path in video_paths:
    # Obtener el nombre base del archivo de video (sin extensión .mp4)
    video_name = os.path.basename(video_path).replace(".mp4", "_r21d.npy")
    
    # Verificar si el archivo .npy correspondiente ya existe
    if video_name not in npy_files:
        videos_to_process.append(video_path)
    else:
        print(f"Características para {video_name} ya existen, saltando...")

# Guardar las rutas filtradas (que aún necesitan procesamiento) en un nuevo archivo
with open("C:/Users/lifeg/OneDrive/Escritorio/Machine/Proyecto_3_Clustering/train_paths_filtered.txt", "w") as f:
    f.write("\n".join(videos_to_process))

print(f"Videos pendientes de procesar guardados en val_paths_filtered.txt. Total: {len(videos_to_process)} videos")
