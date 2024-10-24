import numpy as np

# Cambia 'ruta/al/archivo' por la ruta real del archivo
data = np.load('features_response\\r21d\\r2plus1d_18_16_kinetics\\--33Lscn6sk_000004_000014_r21d.npy')

# Imprime el contenido del archivo
print(data[1])
