import numpy as np


def euclidean_distance(x, y):
    norm_X = np.sum(x ** 2, axis=1)
    norm_Y = np.sum(y ** 2, axis=1)
    scal_prod = x @ y.T
    norm_X = norm_X.reshape(-1, 1)

    return np.sqrt(np.abs(norm_X + norm_Y - 2 * scal_prod))


def cosine_distance(x, y):
    norm_X = np.sum(x ** 2, axis=1)
    norm_Y = np.sum(y ** 2, axis=1)
    scal_prod = x @ y.T
    norm_X = norm_X.reshape(-1, 1)

    return 1 - scal_prod / np.sqrt(norm_X * norm_Y)
