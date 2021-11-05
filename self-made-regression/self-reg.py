import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

class LinearRegression:
    def fit(self, X, y):
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        X = np.array(X)
        X = np.vstack([np.ones(X.shape[0]), X.T]).T
        self.gd(X, y)
        return self
    
    def predict(self, data):
        X = np.array(data)
        X = np.vstack([np.ones(X.shape[0]), X.T]).T
        return np.dot(X, self.theta)

    def score(self, X, y, theta=None, add_ones=False):
        X = np.array(X)
        if add_ones:            
            X = np.vstack([np.ones(X.shape[0]), X.T]).T
        if theta is None:
            return np.sum((np.dot(X, self.theta) - y) ** 2) / (2 * X.shape[0])
        return np.sum((np.dot(X, theta) - y) ** 2) / (2 * X.shape[0])
    
    def gd(self, X, y, learning_rate=0.01):
        self.theta = np.ones(X.shape[1])
        cost_history = [self.score(X, y)]
        delta = learning_rate / X.shape[0] * np.dot(X.T, np.dot(X, self.theta) - y)
        
        while (np.linalg.norm(delta) > 0.0001):
            self.theta -= delta
            cost_history.append(self.score(X, y))
            delta = learning_rate / X.shape[0] * np.dot(X.T, np.dot(X, self.theta) - y)
        
        self.cost_history = np.array(cost_history)
        self.best_score = self.cost_history[-1]
        self.X = X
        self.y = y
    
    def contour_plot(self, r=[[-10, 10], [-10,10]], s=[30, 30], show_best=False):
        r = np.array(r)
        t1 = np.linspace(r[0, 0], r[0, 1], s[0])
        t2 = np.linspace(r[1, 0], r[1, 1], s[1])
        xv, yv = np.meshgrid(t1, t2)
        z = np.zeros(s)

        for i in range(s[0]):
            for j in range(s[1]):
                z[i, j] = self.score(self.X, self.y, theta=[xv[i, j], yv[i, j]])

        plt.contourf(z)
        plt.xlabel('teta0')
        plt.ylabel('teta1')
        if show_best:
            x_best = (self.theta[0] - r[0, 0]) / (r[0, 1] - r[0, 0]) * s[0]
            y_best = (self.theta[1] - r[1, 0]) / (r[1, 1] - r[1, 0]) * s[1]    
            plt.scatter(x_best, y_best, edgecolors='white', s=50)
            plt.annotate(f"{self.theta}", (x_best, y_best), color='white')

def cv_score(X, y, n_splits=10):
    kf = KFold(n_splits=n_splits)
    scores = []
    for ti, vi in kf.split(X, y):    
        X_train = X[ti]
        X_test = X[vi]
        y_train = y[ti]
        y_test = y[vi]
    
        m = LinearRegression()
        m.fit(X_train, y_train)
        scores.append(m.score(X_test, y_test, add_ones=True))
    
    return np.array(scores)