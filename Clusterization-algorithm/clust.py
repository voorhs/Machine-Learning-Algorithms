import numpy as np
from sklearn.datasets import make_blobs

class KMeans:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
    
    def initilize_centroids(self, eur):
        if eur.lower() == "sample":
            self.indeces = np.random.choice(np.arange(self.X.shape[0]), self.n_clusters, replace=False)
            self.centroids = np.copy(self.X[self.indeces])
            self.set_labels()
        elif eur.lower() == "random":
            self.indeces = None
            min_x, min_y = np.min(self.X, axis=0)
            max_x, max_y = np.max(self.X, axis=0)
            y = np.random.rand(self.n_clusters) * (max_y - min_y) + min_y
            x = np.random.rand(self.n_clusters) * (max_x - min_x) + min_x
            self.centroids = np.vstack([x,y]).T
            self.assignment()

    def fit(self, X, eur="sample"):
        self.X = X
        self.initilize_centroids(eur)
        self.labels = self.labels.astype(int)
        i = 0
        while (np.linalg.norm(self.centroids - self.update()) > 0.1):
            self.assignment()
            i += 1
        return self

    def near_ind(self, ind):     
        if ind in self.indeces:
            return ind
        
        res = self.indeces[0]
        for i in self.indeces:
            if np.linalg.norm(self.X[ind] - self.X[i]) < np.linalg.norm(self.X[ind] - self.X[res]):
                res = i
        return res
    
    def set_labels(self):
        self.labels = np.zeros(self.X.shape[0])
        dct = zip(list(self.indeces), list(np.arange(0, self.n_clusters)))
        dct = dict(dct)        
        for i in range(self.X.shape[0]):
            self.labels[i] = dct[self.near_ind(i)]
    
    def near(self, x):
        res = 0
        for i in range(self.centroids.shape[0]):
            if np.linalg.norm(x - self.centroids[i]) < np.linalg.norm(x - self.centroids[res]):
                res = i
        return res

    def assignment(self):
        self.labels = np.zeros(self.X.shape[0])
        for i in range(self.X.shape[0]):
            self.labels[i] = self.near(self.X[i])

    def update(self):
        self.centroids *= 0        
        counts = np.zeros(self.n_clusters)
        for i in range(self.X.shape[0]):            
            self.centroids[self.labels[i]] += self.X[i]
            counts[self.labels[i]] += 1
        for i in range(self.centroids.shape[0]):
            self.centroids[i] /= counts[i]
        return self.centroids

X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
