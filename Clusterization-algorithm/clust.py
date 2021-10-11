from textwrap import indent
import numpy as np

class KMeans:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
    
    def fit(self, X):
        self.X = X
        self.indeces = np.random.choice(np.arange(X.shape[0]), self.n_clusters, replace=False)
        self.centroids = X[self.indeces]
        self.set_labels()
        return self

    def near(self, ind):
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
            self.labels[i] = dct[self.near(i)]
        self.index_to_label = dct