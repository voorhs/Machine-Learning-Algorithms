import numpy as np
from sklearn.datasets import make_blobs

class KMeans:
    def __init__(self, X, n_clusters):
        """
        X is array of points (x, y)
        """

        self.X = X
        self.n_clusters = n_clusters
        self.n_points = X.shape[0]
    
    def init_random(self):
        """
        initializes array of centroids using 'random' strategy:
        random 2dim points from the area of X
        """

        min_x, min_y = np.min(self.X, axis=0)
        max_x, max_y = np.max(self.X, axis=0)

        y_coords = np.random.rand(self.n_clusters) * (max_y - min_y) + min_y
        x_coords = np.random.rand(self.n_clusters) * (max_x - min_x) + min_x

        self.centroids = np.vstack([x_coords,y_coords]).T

    def init_sample(self):
        """
        initializes array of centroids using 'sample' strategy:
        random simple sample from X
        """

        self.indeces = np.random.choice(np.arange(self.X.shape[0]), self.n_clusters, replace=False)
        self.centroids = np.copy(self.X[self.indeces])
    
    def compute_distance_matrix(self, centr_ind):   
        """
        returns matrix 'centr_ind' times 'n_clusters' where
        [i][j] is distance between point X[i] and centroids[j],
        i.e. rho( X[i], centroid[j] )
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
        self.centroids = np.zeros((self.n_clusters, 2))

        t = np.random.randint(0, self.n_points, size=1)
        self.indeces[0] = t
        self.centroids[0] = self.X[t]

        for i in range(1, self.n_clusters):
            t = np.argmax(self.compute_distance_matrix(i).mean(axis=1))
            self.indeces[i] = t
            self.centroids[i] = self.X[t]

    def init_centr(self, heur):
        """
        initializes centroids with given heuristic
        """
        dct = {
            "sample": self.init_sample,
            "random": self.init_random,
            "distant": self.init_distant,
        }
        dct[heur]()
    
    def near_center(self, point):
        """
        returns the index of the nearest centroids from given point
        """
        res = 0
        for i in range(self.n_clusters):
            if np.linalg.norm(point - self.centroids[i]) < np.linalg.norm(point - self.centroids[res]):
                res = i
        return res

    def set_labels(self):
        """
        sets labels of membership in the cluster
        """
        self.labels = np.zeros(self.n_points)
        for i in range(self.n_points):
            self.labels[i] = self.near_center(self.X[i])  

    def update_centers(self):
        """
        compute new centroids for every cluster
        """
        self.centroids = np.zeros((self.n_clusters, 2))
        for i in range(self.n_clusters):
            self.centroids[i] = self.X[self.labels == i].mean(axis=0)
        return self.centroids
    
    def fit(self, heur="random", prec=0.001):
        """
        main function that launches centroids and labels initialization
        and executes iterative k-means algorithm
        """
        self.init_centr(heur)
        self.set_labels()

        while(np.linalg.norm(self.centroids - self.update_centers()) > prec):
            self.set_labels()
        
        return self
