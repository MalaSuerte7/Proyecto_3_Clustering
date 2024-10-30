import numpy as np
import math

class DBSCAN:
    UNCLASSIFIED = -1
    NOISE = None

    def __init__(self, data, eps=0.5, min_points=2):
        self.data = data
        self.eps = eps
        self.min_points = min_points
        self.labels = [self.UNCLASSIFIED] * len(data)
    
    # Calcula la distancia euclidiana
    def euclidean_dist(self, p, q):
        return math.sqrt(np.power(p - q, 2).sum())

    # Verifica si un punto está dentro del radio eps
    def eps_neighborhood(self, p, q):
        return self.euclidean_dist(p, q) < self.eps

    # Encuentra los vecinos dentro de la distancia eps
    def region_query(self, point_id):
        neighbors = []
        for i in range(len(self.data)):
            if self.eps_neighborhood(self.data[point_id], self.data[i]):
                neighbors.append(i)
        return neighbors

    # Expande el cluster desde un punto dado
    def expand_cluster(self, point_id, cluster_id):
        # Encuentra los vecinos iniciales de point_id
        seeds = self.region_query(point_id)
        
        # Verifica si el punto es ruido (no cumple con min_points)
        if len(seeds) < self.min_points:
            self.labels[point_id] = self.NOISE
            return False
        else:
            # Asigna el cluster_id al punto y a sus vecinos iniciales
            self.labels[point_id] = cluster_id
            for seed_id in seeds:
                self.labels[seed_id] = cluster_id
            
            # Expande el cluster usando la lista de semillas
            i = 0
            while i < len(seeds):
                current_point = seeds[i]
                results = self.region_query(current_point)
                
                # Si current_point es un punto central, expandimos el cluster
                if len(results) >= self.min_points:
                    for result_point in results:
                        if self.labels[result_point] in [self.UNCLASSIFIED, self.NOISE]:
                            if self.labels[result_point] == self.UNCLASSIFIED:
                                seeds.append(result_point)
                            self.labels[result_point] = cluster_id
                i += 1
            return True
    
    # Método principal de DBSCAN para ajustar el modelo
    def fit(self):
        cluster_id = 1
        for point_id in range(len(self.data)):
            # Si el punto no está clasificado, intenta expandir un nuevo cluster
            if self.labels[point_id] == self.UNCLASSIFIED:
                if self.expand_cluster(point_id, cluster_id):
                    cluster_id += 1
        return self.labels

    # Predice el cluster de nuevos puntos de datos (opcional)
    def predict(self, new_data):
        predictions = []
        for point in new_data:
            distances = [self.euclidean_dist(point, self.data[center_id]) for center_id, label in enumerate(self.labels) if label not in [self.NOISE, self.UNCLASSIFIED]]
            closest_cluster = self.labels[np.argmin(distances)]
            predictions.append(closest_cluster if closest_cluster is not self.NOISE else self.NOISE)
        return predictions

# # Ejemplo de prueba (opcional)
# def test_dbscan():
#     data = np.array([[1, 1.1], [1.2, 0.8], [0.8, 1], [3.7, 4], [3.9, 3.9], [3.6, 4.1], [10, 10]])
#     model = DBSCAN(data, eps=0.5, min_points=2)
#     print("Etiquetas de cluster:", model.fit())

# test_dbscan()

# Inpirado de https://scrunts23.medium.com/dbscan-algorithm-from-scratch-in-python-475b82e0571c
# y de https://github.com/choffstein/dbscan/blob/master/dbscan/dbscan.py
# y de la pagina 67 del pdf _Chapter 4.1_ Clustering.pdf del curso