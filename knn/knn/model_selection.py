from collections import defaultdict

import numpy as np

from sklearn.model_selection import KFold, BaseCrossValidator
from sklearn.metrics import accuracy_score

from knn.classification import BatchedKNNClassifier


def knn_cross_val_score(X, y, k_list, scoring, cv=None, **kwargs):
    y = np.asarray(y)

    if scoring == "accuracy":
        scorer = accuracy_score
    else:
        raise ValueError("Unknown scoring metric", scoring)

    if cv is None:
        cv = KFold(n_splits=5)
    elif not isinstance(cv, BaseCrossValidator):
        raise TypeError("cv should be BaseCrossValidator instance", type(cv))

    ans = defaultdict(list)
    kmax = max(k_list)
    model = BatchedKNNClassifier(kmax, **kwargs)

    for train_index, test_index in cv.split(X):
        model.fit(X[train_index], y[train_index])
        k_distances, k_indices = model.kneighbors(X[test_index], True)

        for k in k_list:
            y_pred = model._predict_precomputed(
                k_indices[:, :k], k_distances[:, :k])
            cur = scorer(y[test_index], y_pred)
            ans[k].append(cur)

    for k, val in ans.items():
        ans[k] = np.array(val)

    return ans
