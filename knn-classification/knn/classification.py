import numpy as np

from sklearn.neighbors import NearestNeighbors
from knn.nearest_neighbors import NearestNeighborsFinder


class KNNClassifier:
    EPS = 1e-5

    def __init__(self, n_neighbors, algorithm='my_own', metric='euclidean', weights='uniform'):
        if algorithm == 'my_own':
            finder = NearestNeighborsFinder(
                n_neighbors=n_neighbors, metric=metric)
        elif algorithm in ('brute', 'ball_tree', 'kd_tree',):
            finder = NearestNeighbors(
                n_neighbors=n_neighbors, algorithm=algorithm, metric=metric)
        else:
            raise ValueError("Algorithm is not supported", metric)

        if weights not in ('uniform', 'distance'):
            raise ValueError("Weighted algorithm is not supported", weights)

        self._finder = finder
        self._weights = weights

    def fit(self, X, y=None):
        self._finder.fit(X)
        self._labels = np.asarray(y)
        return self

    def _predict_precomputed(self, indices, distances):
        if self._weights == 'distance':
            weights = 1 / (distances + self.EPS)
        elif self._weights == 'uniform':
            weights = np.ones_like(distances)
        else:
            raise ValueError("Weighted algorithm is not supported", weights)

        y = self._labels[indices]
        N = self._labels.size
        y += (N * np.arange(y.shape[0]))[:, None]
        ans = np.bincount(y.ravel(), minlength=N * y.shape[0],
                          weights=weights.ravel()
                          ).reshape(-1, N)

        return ans.argmax(axis=1)

    def kneighbors(self, X, return_distance=False):
        return self._finder.kneighbors(X, return_distance=return_distance)

    def predict(self, X):
        distances, indices = self.kneighbors(X, return_distance=True)
        return self._predict_precomputed(indices, distances)


class BatchedMixin:
    def __init__(self):
        self._batch_size = None

    def set_batch_size(self, batch_size):
        self._batch_size = batch_size


class BatchedKNNClassifier(KNNClassifier, BatchedMixin):
    '''
    Нам нужен этот класс, потому что мы хотим поддержку обработки батчами
    в том числе для классов поиска соседей из sklearn
    '''

    def __init__(self, n_neighbors, algorithm='my_own', metric='euclidean', weights='uniform', batch_size=None):
        KNNClassifier.__init__(
            self,
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            weights=weights,
            metric=metric,
        )
        self.set_batch_size(batch_size)

    def kneighbors(self, X, return_distance=False):
        if self._batch_size is None or self._batch_size >= X.shape[0]:
            return super().kneighbors(X, return_distance=return_distance)

        split_indices = [index for index in range(
            self._batch_size, X.shape[0], self._batch_size)]
        batches = np.vsplit(X, split_indices)

        distances = []
        indices = []
        for batch in batches:
            dist, ind = self._finder.kneighbors(batch, return_distance=True)
            distances.append(dist)
            indices.append(ind)

        if return_distance:
            return np.vstack(distances), np.vstack(indices)

        return np.vstack(indices)

    # def predict(self, X):
    #     if self._batch_size is None or self._batch_size >= X.shape[0]:
    #         return super().predict(X)

    #     distances, indices = self.kneighbors(X, return_distance=True)

    #     split_indices = [index for index in range(
    #         self._batch_size, X.shape[0], self._batch_size)]

    #     dists = np.vsplit(distances, split_indices)
    #     inds = np.vsplit(indices, split_indices)

    #     y = []
    #     for ind, dist in zip(inds, dists):
    #         y.append(self._predict_precomputed(ind, dist))

    #     return np.concatenate(y)
