import numpy as np
from numpy.lib.function_base import diff
from sklearn.datasets import make_blobs

class KMeans:
    def __init__(self, X, n_clusters, n_features):
        """
        X is array of points (x, y)
        """

        self.X = X
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.n_points = X.shape[0]
    
    def init_random(self):
        """
        initializes array of centroids using 'random' strategy:
        random points from the area of X
        """

        min_coords = np.min(self.X, axis=0)
        max_coords = np.max(self.X, axis=0)

        coords = np.zeros((self.n_features, self.n_clusters))
        for i in range(self.n_features):
            coords[i] = np.random.rand(self.n_clusters) * (max_coords[i] - min_coords[i]) + min_coords[i]

        self.centroids = coords.T

    def init_sample(self):
        """
        initializes array of centroids using 'sample' strategy:
        random simple sample from X
        """

        self.indeces = np.random.choice(np.arange(self.X.shape[0]), self.n_clusters, replace=False)
        self.centroids = np.copy(self.X[self.indeces])
    
    def compute_distance_matrix(self, centr_ind):   
        """
        высчитывает и возвращает матрицу расстояний от всех точек до центроидов,
        построенных 'centr_int-1` центроидов.
        нужно для реализации `distant`-стратегии инициализации центроидов
        """     
        res = np.zeros((self.n_points, centr_ind))
        for i in range(0, self.n_points):
            for j in range(0, centr_ind):
                res[i][j] = np.linalg.norm(self.centroids[j] - self.X[i])
        return res

    def init_distant(self):
        """
        initializes array of centroids using 'distant' strategy:
        1st centroid is random point of X, 2nd is the farthest point of X from 1st centroid,
        3rd is the farthest point of X from 1st ans 2nd centroids etc.
        """
        self.indeces = np.zeros(self.n_clusters, dtype=int)
        self.centroids = np.zeros((self.n_clusters, self.n_features))

        t = np.random.randint(0, self.n_points, size=1)
        self.indeces[0] = t
        self.centroids[0] = self.X[t]

        for i in range(1, self.n_clusters):
            t = np.argmax(self.compute_distance_matrix(i).mean(axis=1))
            self.indeces[i] = t
            self.centroids[i] = self.X[t]

    def init_centr(self, heur):
        """
        initializes centroids by given heuristic
        """
        dct = {
            "sample": self.init_sample,
            "random": self.init_random,
            "distant": self.init_distant,
        }
        dct[heur]()
    
    def near_center(self, point):
        """
        returns the index of the nearest centroid from given point
        """
        res = 0
        for i in range(self.n_clusters):
            if np.linalg.norm(point - self.centroids[i]) < np.linalg.norm(point - self.centroids[res]):
                res = i
        return res

    def set_labels(self):
        """
        marks points from X in accordance with which centroid is closer
        """
        self.labels = np.zeros(self.n_points)
        for i in range(self.n_points):
            self.labels[i] = self.near_center(self.X[i])  

    def update_centers(self):
        """
        updates centroids for every cluster
        """
        self.centroids = np.zeros((self.n_clusters, self.n_features))
        for i in range(self.n_clusters):
            self.centroids[i] = self.X[self.labels == i].mean(axis=0)
        return self.centroids
    
    def perform(self, heur, prec):
        self.init_centr(heur)
        self.set_labels()

        while(np.linalg.norm(self.centroids - self.update_centers()) > prec):
            self.set_labels()
        
        self.distances = self.compute_distance_matrix(self.n_clusters)
        self.compute_SSW()
        self.save_config()

        return self
    
    def save_config(self):
        self.configs.append([self.centroids, self.labels, self.indeces, self.distances, self.SSW])

    def find_best_norm(self, n_start, prec):
        min_counter = n_start
        res = 0
        for i in range(n_start):
            counter = 0
            for j in range(n_start):
                if np.linalg.norm(self.configs[i][0] - self.configs[j][0]) > 0.1:
                    counter += 1
            if counter < min_counter:
                res = i
                min_counter = counter
        
        return self.configs[res]
    
    def compute_SSW(self):
        cluster_squares = self.distances ** 2

        self.within_cluster_squares = np.zeros(self.n_clusters)
        for i in range(self.n_clusters):
                self.within_cluster_squares[i] = np.sum(cluster_squares[self.labels == i], axis=0)[i]

        self.SSW = np.sum(self.within_cluster_squares)
    
    def find_best_var(self, n_start):
        SSW_values = np.zeros(n_start)
        for i in range(n_start):
            SSW_values[i] = self.configs[i][4]
        
        return self.configs[np.argmin(SSW_values)]

    def fit(self, heur="random", n_start=3, prec_kmeans=0.001, prec_find_best=0.1):
        """
        main function that calls centroids initialization
        and performs iterative k-means algorithm
        """
        self.configs = []
        for i in range(n_start):
            self.perform(heur, prec_kmeans)
        
        best_config = self.find_best_var(n_start)
        self.centroids = best_config[0]
        self.labels = best_config[1]
        self.indeces = best_config[2]
        self.distances = best_config[3]

        return self

class elbow:
    def __init__(self, X, n_features, heur='sample', kmax=10, n_start=3):
        self.kmax = kmax
        self.variances_for_k = np.zeros(kmax)
        self.models = []
        for k in range(kmax):
            m = KMeans(X, n_clusters=k+1, n_features=n_features).fit(heur=heur, n_start=n_start)
            self.variances_for_k[k] = m.SSW
            self.models.append(m)
        
    def choose_k(self):
        diffrences = np.zeros(self.kmax)
        for k in range(1, self.kmax):
            diffrences[k] = self.variances_for_k[k - 1] - self.variances_for_k[k]
        
        proportions = np.zeros(self.kmax)
        for k in range(1, self.kmax):
            proportions[k] = diffrences[k] / self.variances_for_k[k]
        
        
        return np.argmin(proportions[1:]) + 1