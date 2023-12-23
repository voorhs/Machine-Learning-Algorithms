import numpy as np
from numpy.lib.function_base import gradient
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

class LogisticRegression:
    def fit(self, X, y, regul=0, feat_mapping=False, max_exp=None):
        self.n_samples = X.shape[0]
        self.feat_mapping = False
        X = np.array(X)

        if feat_mapping:
            X = feature_mapping(X, max_exp)            
            self.feat_mapping = True
            self.max_exp = max_exp
        else:
            X = np.vstack([np.ones(X.shape[0]), X.T]).T
        
        self.n_features = X.shape[1]
        
        self.gd(X, y, regul)
        return self
    
    def predict(self, data):
        return self.hypothesis(data, self.theta)

    def hypothesis(self, x, theta):
        return 1 / (1 + np.exp(-np.dot(x, theta)))
    
    def gradient(self, X, y, theta, regul):
        mask = np.ones_like(theta)
        mask[0] = 0
        return np.dot(X.T, self.hypothesis(X, theta) - y) / X.shape[0] + regul / X.shape[0] * theta * mask

    def cost_function(self, X, y, theta, regul):
        m = X.shape[0]
        total_cost = -(1 / m) * np.sum(y * np.log(self.hypothesis(X, theta)) + (1 - y) * np.log(1 - self.hypothesis(X, theta))) + regul * np.sum(theta ** 2) / (2 * X.shape[0])
        return total_cost

    def score(self, X, y, regul=0, theta=None, add_ones=False):
        X = np.array(X)
        y = np.array(y)

        if self.feat_mapping:
            X = feature_mapping(X, self.max_exp)
        elif add_ones:
            X = np.vstack([np.ones(X.shape[0]), X.T]).T

        if theta is None:
            return self.cost_function(X, y, self.theta, regul)
        
        return self.cost_function(X, y, np.array(theta), regul)
    
    def gd(self, X, y, regul, learning_rate=0.03):
        self.theta = np.zeros(X.shape[1])
        cost_history = [self.score(X, y, regul)]
        delta = learning_rate * self.gradient(X, y, self.theta, regul)
        
        while (np.linalg.norm(delta) > 0.0001):
            self.theta -= delta
            cost_history.append(self.score(X, y, regul))
            delta = learning_rate * self.gradient(X, y, self.theta, regul)
        
        self.cost_history = np.array(cost_history)
        self.best_score = self.cost_history[-1]
        self.X = X
        self.y = y
    
def feature_mapping(X, max_exp):
        X = np.array(X)
        n_features = (max_exp + 2) * (max_exp + 1) // 2
        X_new = np.zeros((n_features, X.shape[0]))
        k = 0
        for i in range(max_exp + 1):
            for j in range(max_exp + 1 - i):
                new_feature = X[:, 0] ** i * X[:, 1] ** j
                X_new[k] = new_feature
                k += 1
        return X_new.T

def contour_plot(model, max_exp, data, s=[30, 30]):
    r = np.zeros((2, 2))
    r[0, 0] = data[:, 0].min()
    r[0, 1] = data[:, 0].max()
    r[1, 0] = data[:, 1].min()
    r[1, 1] = data[:, 1].max()
    t1 = np.linspace(r[0, 0], r[0, 1], s[0])
    t2 = np.linspace(r[1, 0], r[1, 1], s[1])
    xv, yv = np.meshgrid(t1, t2)
    z = np.zeros(s)

    for i in range(s[0]):
        for j in range(s[1]):
            feats = np.array([xv[i, j], yv[i, j]]).reshape(1, 2)
            z[i, j] = model.predict(feature_mapping(feats, max_exp))

    plt.contourf(z)
    plt.xlabel('x1')
    plt.ylabel('x2')
        

def cv_score(X, y, n_splits=10, regul=0, feat_mapping=False, max_exp=None):
    kf = KFold(n_splits=n_splits)
    scores = []
    for ti, vi in kf.split(X, y):    
        X_train = X[ti]
        X_test = X[vi]
        y_train = y[ti]
        y_test = y[vi]
    
        m = LogisticRegression()
        m.fit(X_train, y_train, regul, feat_mapping, max_exp)
        scores.append(m.score(X_test, y_test, add_ones=True))
    
    return np.array(scores)