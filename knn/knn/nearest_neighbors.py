import numpy as np

from knn.distances import euclidean_distance, cosine_distance


def get_best_ranks(ranks, top, axis=1, return_ranks=False):
    raise NotImplementedError()


class NearestNeighborsFinder:
    def __init__(self, n_neighbors, metric="euclidean"):
        self.n_neighbors = n_neighbors

        if metric == "euclidean":
            self._metric_func = euclidean_distance
        elif metric == "cosine":
            self._metric_func = cosine_distance
        else:
            raise ValueError("Metric is not supported", metric)
        self.metric = metric

    def fit(self, X, y=None):
        self._X = X
        return self

    def kneighbors(self, X, return_distance=False):
        pairwise_distance = self._metric_func(X, self._X)

        k_indices = np.argpartition(
            pairwise_distance, self.n_neighbors-1, axis=1
        )[:, :self.n_neighbors]

        k_distances = np.take_along_axis(pairwise_distance, k_indices, axis=1)

        change = k_distances.argsort(axis=1)

        k_indices = np.take_along_axis(k_indices, change, axis=1)
        k_distances = np.take_along_axis(k_distances, change, axis=1)

        if return_distance:
            return k_distances, k_indices
        return k_indices
